import os
import psycopg2
import urllib.parse
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.chat_models import ChatOllama # NEU: Für lokale Modelle
from flashrank import Ranker, RerankRequest

# SETUP
load_dotenv()

def load_models():
    # Cache-Dir für Docker optimiert
    ranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir="/app/models/flashrank")
    
    # LLM: OLLAMA 
    llm = ChatOllama(model="qwen2.5:32b", base_url="http://ollama:11434", temperature=0.1,num_ctx=32768)
    extraction_llm = ChatOllama(model="qwen2.5:32b", base_url="http://ollama:11434", temperature=0,num_ctx=8192)
    
    # LLM: OPENAI 
    #llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    #extraction_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # EMBEDDINGS: OLLAMA LOKAL  
    # Nutzt mxbai-embed-large oder nomic-embed-text (beide sehr gut für RAG)
    from langchain_community.embeddings import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://ollama:11434",num_ctx=8192)

    # EMBEDDINGS: OPENAI 
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    return ranker, llm, extraction_llm, embeddings

ranker, llm, extraction_llm, embeddings_model = load_models()

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT")
    )

def extract_entities_universal(question):
    prompt = f"""Extrahiere die Hauptsubjekte (Personen, Firmen, Organisationen) aus der Frage. 
    Nenne für Organisationen auch bekannte Kurzformen oder alternative Bezeichnungen.
    Frage: '{question}'
    Antworte NUR als kommagetrennte Liste ohne Einleitung."""
    try:
        res = extraction_llm.invoke(prompt)
        # res.content funktioniert bei Ollama genau wie bei OpenAI
        return [e.strip() for e in res.content.split(",") if e.strip()]
    except Exception as e:
        print(f"⚠️ Fehler bei Extraktion: {e}")
        return []

def search_hybrid_graph(question):
    # 1. SCHLAGWORTE EXTRAHIEREN (Wichtig für die Suche!)
    # Wir nehmen die Wörter aus der Frage, die länger als 4 Buchstaben sind
    search_terms = [w.strip("?!.,").lower() for w in question.split() if len(w) > 4]
    # Falls die Frage sehr kurz ist, nehmen wir alles
    if not search_terms:
        search_terms = [question.lower()]
    
    # Bereite die SQL-Suche vor: ["%ersthelfer%", "%schnoor%", ...]
    sql_keywords = [f"%{t}%" for t in search_terms]

    # 2. Embedding für die semantische Suche
    query_vector = embeddings_model.embed_query(question)
    
    conn = get_connection()
    cur = conn.cursor()

    # STUFE 1: Hybrid-Suche (Vektor + Keyword ANY Boost)
    # Wir suchen jetzt nach Chunks, die IRGENDEINES der Schlagworte enthalten
    cur.execute("""
        WITH vector_search AS (
            SELECT 
                p.full_text as parent_text, 
                p.title, 
                p.source_url, 
                p.id as p_id, 
                c.content, 
                (1 - (c.embedding <=> %s::vector)) as similarity
            FROM document_chunks c
            JOIN parent_documents p ON c.parent_id = p.id
            ORDER BY c.embedding <=> %s::vector
            LIMIT 60 
        ),
        keyword_search AS (
            SELECT 
                p.full_text as parent_text, 
                p.title, 
                p.source_url, 
                p.id as p_id, 
                c.content,
                1.0 as similarity 
            FROM document_chunks c
            JOIN parent_documents p ON c.parent_id = p.id
            WHERE c.content ILIKE ANY(%s)  -- FINDET JETZT "Ersthelfer" sicher!
               OR p.full_text ILIKE ANY(%s)
            LIMIT 30
        )
        SELECT * FROM vector_search
        UNION ALL
        SELECT * FROM keyword_search
    """, (query_vector, query_vector, sql_keywords, sql_keywords))
    
    rows = cur.fetchall()

    # Passages für den Reranker aufbereiten
    passages = []
    for r in rows:
        passages.append({
            "id": str(len(passages)),
            "text": r[4], # c.content
            "meta": {
                "title": r[1],
                "url": r[2],
                "p_id": str(r[3]),
                "parent_text": r[0]
            }
        })

    if not passages:
        cur.close()
        conn.close()
        return "Keine relevanten Dokumente gefunden.", "Keine Graph-Daten."

    # STUFE 2: Re-Ranking (Der Reranker entscheidet, was wirklich wichtig ist)
    rerank_results = ranker.rerank(RerankRequest(query=question, passages=passages))
    top_results = rerank_results[:8] # Wir nehmen 8 für mehr Sicherheit

    context_text = ""
    doc_ids = []
    for res in top_results:
        meta = res['meta']
        safe_url = urllib.parse.quote(meta['url'], safe=':/?&=')
        doc_ids.append(meta['p_id'])
        context_text += f"\n--- QUELLE: {meta['title']} ---\nURL: {safe_url}\nINHALT:\n{meta['parent_text']}\n"

    # STUFE 3: Graph-Scan (Bleibt wie gehabt)
    search_terms_graph = extract_entities_universal(question)
    sql_terms_graph = [f"%{t}%" for t in search_terms_graph] if search_terms_graph else sql_keywords
    
    graph_knowledge = ""
    graph_query = """
    SELECT DISTINCT n1.entity_name, e.relation_type, n2.entity_name, n2.entity_type
    FROM document_edges e
    JOIN document_nodes n1 ON e.source_node_id = n1.id
    JOIN document_nodes n2 ON e.target_node_id = n2.id
    WHERE (
        e.source_doc_id = ANY(%s::uuid[]) 
        OR n1.entity_name ILIKE ANY(%s) 
        OR n2.entity_name ILIKE ANY(%s)
    )
    AND e.confidence >= 3
    LIMIT 30;
    """
    cur.execute(graph_query, (doc_ids, sql_terms_graph, sql_terms_graph))
    triples = cur.fetchall()
    
    if triples:
        for s, p, o, o_type in triples:
            graph_knowledge += f"- {s} {p} {o} (Typ: {o_type})\n"
    else:
        graph_knowledge = "Keine spezifischen Graph-Verknüpfungen gefunden."

    cur.close()
    conn.close()
    return context_text, graph_knowledge

