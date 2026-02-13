import os
import uuid
import json
import psycopg2
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# 1. SETUP
load_dotenv()

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".html", 
    ".asc", ".md", ".txt", ".png", ".jpg", ".jpeg", ".tiff"
}

# Basis-URL für den späteren Zugriff über WebUI (Standardmäßig localhost für Tests)
# In der .env Datei kannst du FILE_SERVER_URL=https://dein-server.de/files/ setzen
FILE_SERVER_BASE_URL = os.getenv("FILE_SERVER_URL", "http://localhost:8000/files/")

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT")
    )

# KI-Modelle
# OPENAI 
#embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# OLLAMA LOKAL 
embeddings_model = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://ollama:11434") # "ollama" ist der Name des Containers im Netzwerk

llm = ChatOllama(model="qwen2.5:32b",base_url="http://ollama:11434", temperature=0,num_ctx=16384)

# Hilfsfunktion für globale Konsistenz (Gedächtnis für Entitäts-Typen)
def get_existing_types(cur):
    try:
        cur.execute("SELECT DISTINCT entity_type FROM document_nodes")
        return [row[0] for row in cur.fetchall()]
    except:
        return []

# 2. UNIVERSAL GRAPH EXTRACTION
def extract_graph_triples(text, existing_types=[]):
    type_hint = f"Bisher bekannte Typen: {', '.join(existing_types)}" if existing_types else ""
    
    prompt = f"""
    Extrahiere nur signifikante Informationen als Wissens-Tripel (Subjekt | Beziehung | Objekt).

    ### QUALITÄTS-REGELN ###
    1. KEINE PRONOMEN: Nutze niemals "er", "sie", "es" oder "dieser" als Subjekt. Ersetze sie durch den Namen der Entität.
    2. ATOMARE FAKTEN: Ein Tripel muss ohne den restlichen Text Sinn ergeben. (Schlecht: "Er | hat | 5" -> Gut: "Projekt X | Risikostufe | 5").
    3. RELEVANZ-FILTER: Ignoriere Füllwörter oder triviale Sätze. Extrahiere nur harte Fakten wie z.B., technische Daten, Personen, Projektnamen oder Projektstatus.
    4. KONSISTENZ: Nutze für  Dinge gleiche Dinge immer exakt den gleichen Namen. 
    5. BEWERTUNG: Weise jedem Tripel einen Confidence-Score zwischen 1 und 5 zu (5 = absolut sicher aus dem Text belegbar, 1 = vage Vermutung).

    {type_hint}

    Antworte NUR im JSON-Format:
    {{"triples": [
        {{"s": "Entität", "s_type": "TYP", "p": "Beziehung", "o": "Wert/Objekt", "o_type": "TYP", "confidence": 5}}
    ]}}
    
    Text: {text}
    """ 
    try:
        response = llm.invoke(prompt)
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_content)
        return data.get("triples", [])
    except Exception as e:
        print(f"   ! Fehler bei Extraktion: {e}")
        return []

def save_to_graph(cur, triples, doc_id):
    """Speichert nur qualitativ hochwertige Knoten und Kanten in den Graphen."""
    for t in triples:
        # 1. PRÜFUNG: Vollständigkeit und Confidence
        # Wir erwarten jetzt auch ein Feld 'confidence' vom LLM
        if not all(k in t for k in ['s', 'p', 'o']):
            continue
        
        # Filter: Nur Fakten mit hoher Sicherheit (falls vom LLM geliefert, sonst Default 5)
        if int(t.get('confidence', 5)) < 4:
            continue

        # 2. QUALITÄTS-FILTER: Längenbeschränkung
        # Verhindert, dass ganze Sätze als "Entität" im Graphen landen
        s_val = str(t['s']).strip()
        p_val = str(t['p']).strip()
        o_val = str(t['o']).strip()

        if len(s_val) > 60 or len(o_val) > 100 or len(p_val) > 50:
            # Zu lange Texte sind meistens kein sauberer Wissens-Fakt
            continue
            
        if len(s_val) < 2 or len(o_val) < 2:
            # Zu kurze Fragmente (z.B. "X", "5") ohne Kontext ignorieren
            continue

        # 3. NORMALISIERUNG
        # Wir vereinheitlichen die Beziehung (Kleinschreibung), um Duplikate zu vermeiden
        relation = p_val.lower()
        
        entities = [
            (s_val, t.get('s_type', 'UNKNOWN').upper()), 
            (o_val, t.get('o_type', 'UNKNOWN').upper())
        ]
        
        node_ids = []
        for name, e_type in entities:
            # Eindeutige ID über kleingeschriebenen Namen generieren 
            clean_name = name.strip()
            e_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, clean_name.lower()))
            node_ids.append(e_id)
            
            cur.execute("""
                INSERT INTO document_nodes (id, entity_name, entity_type) 
                VALUES (%s, %s, %s) 
                ON CONFLICT (id) DO UPDATE SET 
                    entity_type = EXCLUDED.entity_type 
                WHERE EXCLUDED.entity_type != 'UNKNOWN'
            """, (e_id, clean_name, e_type))
        
        # 4. KANTE SPEICHERN
        cur.execute("""
            INSERT INTO document_edges (id, source_node_id, target_node_id, relation_type, source_doc_id, confidence)
            VALUES (%s, %s, %s, %s, %s, %s) 
            ON CONFLICT DO NOTHING
        """, (str(uuid.uuid4()), node_ids[0], node_ids[1], relation, doc_id, int(t.get('confidence', 5))))

# 3. VERARBEITUNGSLOGIK
def ingest_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS: return

    filename = os.path.basename(file_path)
    print(f"\n🚀 Verarbeite Dokument: {filename}")
    
    try:
        # A. Konvertierung mit Docling (behält Tabellenstrukturen bei)
        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()
        
        # B. Chunking
        # Große Chunks (Parent) für Kontext, kleine (Child) für präzise Vektorsuche
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1750, chunk_overlap=250)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        
        conn = get_connection()
        cur = conn.cursor()
        
        # Bekannte Typen für Prompt-Konsistenz laden
        known_types = get_existing_types(cur)

        parent_chunks = parent_splitter.split_text(markdown_text)
        
        # Web-URL für den Zugriff generieren (Server-ready)
        # Beispiel: http://localhost:8000/files/Zeugnis.pdf
        web_url = f"{FILE_SERVER_BASE_URL}{filename}"
        
        for idx, p_text in enumerate(parent_chunks):
            p_id = str(uuid.uuid4())
            
            # 1. Parent speichern (Inklusive der URL für klickbare Links in WebUI)
            cur.execute("""
                INSERT INTO parent_documents (id, title, full_text, source_url) 
                VALUES (%s, %s, %s, %s)
            """, (p_id, filename, p_text, web_url))
            
            # 2. Graph-Fakten extrahieren (On-the-fly Schema Lernen)
            print(f"   [Chunk {idx+1}/{len(parent_chunks)}] Extrahiere Fakten...")
            triples = extract_graph_triples(p_text, existing_types=known_types)
            save_to_graph(cur, triples, p_id)
            
            # 3. Vektor-Speicherung (Embeddings für Ähnlichkeitssuche)
            child_chunks = child_splitter.split_text(p_text)
            if child_chunks:
                vectors = embeddings_model.embed_documents(child_chunks)
                for c_text, vec in zip(child_chunks, vectors):
                    cur.execute("""
                        INSERT INTO document_chunks (id, parent_id, content, embedding) 
                        VALUES (%s, %s, %s, %s)
                    """, (str(uuid.uuid4()), p_id, c_text, vec))
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Fertig: {filename}")
        
    except Exception as e:
        print(f"❌ Schwerer Fehler bei {file_path}: {e}")

# 4. STARTPUNKT (Inkrementelles Update)
if __name__ == "__main__":
    doc_dir = os.getenv("DOC_DIR")
    if not doc_dir or not os.path.exists(doc_dir):
        print("Fehler: DOC_DIR nicht konfiguriert.")
    else:
        try:
            conn = get_connection()
            cur = conn.cursor()
            # Nur Dateien laden, die noch nicht in der Datenbank existieren
            cur.execute("SELECT DISTINCT title FROM parent_documents")
            indexed_files = {row[0] for row in cur.fetchall()}
            cur.close()
            conn.close()
            print(f"Status: {len(indexed_files)} Dokumente bereits indexiert.")
        except:
            indexed_files = set()

        files = [f for f in os.listdir(doc_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
        
        for filename in files:
            if filename in indexed_files:
                print(f"⏩ Überspringe: {filename} (Bereits vorhanden)")
                continue
            ingest_document(os.path.join(doc_dir, filename))
