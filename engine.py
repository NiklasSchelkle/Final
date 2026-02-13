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
    query_vector = embeddings_model.embed_query(question)
    conn = get_connection()
    cur = conn.cursor()

    # STUFE 1: Vektor-Suche
    cur.execute("""
        SELECT p.full_text, p.title, p.source_url, p.id, c.content, p.created_at
        FROM document_chunks c
        JOIN parent_documents p ON c.parent_id = p.id
        ORDER BY (c.embedding <=> %s::vector) ASC, p.created_at DESC
        LIMIT 20;
    """, (query_vector,))
    rows = cur.fetchall()
    
    if not rows:
        cur.close()
        conn.close()
        return "", "Keine Daten gefunden."

    # STUFE 2: Re-Ranking
    passages = [{"id": i, "text": r[4], "meta": {"title": r[1], "url": r[2], "parent_text": r[0], "p_id": r[3]}} for i, r in enumerate(rows)]
    rerank_results = ranker.rerank(RerankRequest(query=question, passages=passages))
    top_results = rerank_results[:6]

    context_text = ""
    doc_ids = []
    for res in top_results:
        meta = res['meta']
        safe_url = urllib.parse.quote(meta['url'], safe=':/?&=')
        doc_ids.append(meta['p_id'])
        context_text += f"\n--- QUELLE: {meta['title']} ---\nURL: {safe_url}\nINHALT:\n{meta['parent_text']}\n"

    # STUFE 3: Graph-Scan 
    graph_knowledge = ""
    search_terms = extract_entities_universal(question)

    # Suchbegriffe für SQL
    sql_terms = [f"%{t}%" for t in search_terms]

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
    AND e.confidence >= 4  -- Nur hochwertige Fakten!
    LIMIT 20; -- Reduziert von 65 auf 20
    """
    cur.execute(graph_query, (doc_ids, sql_terms, sql_terms))
    triples = cur.fetchall()
    
    print(f"🔍 Graph-Scan abgeschlossen. Gefundene Tripel: {len(triples)}")

    if triples:
        for s, p, o, o_type in triples:
            graph_knowledge += f"- {s} {p} {o} (Typ: {o_type})\n"
    else:
        graph_knowledge = "Keine spezifischen Graph-Verknüpfungen gefunden."

    cur.close()
    conn.close()
    return context_text, graph_knowledge
