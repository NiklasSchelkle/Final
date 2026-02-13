from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Any
import uvicorn
from engine import search_hybrid_graph, llm 

app = FastAPI(title="SCHNOOR Hybrid RAG API")

class ChatQuery(BaseModel):
    question: Any # Erlaubt Text oder die Liste aus der Open WebUI Pipe

@app.post("/query")
async def handle_query(query: ChatQuery):
    # 1. DATEN SORTIEREN
    if isinstance(query.question, list):
        messages = query.question
        last_user_message = messages[-1]["content"] if messages else ""
    else:
        messages = []
        last_user_message = query.question

    # 2. QUERY REWRITING
    search_query = last_user_message
    if len(messages) > 1:
        history_context = ""
        for msg in messages[-3:-1]:
            history_context += f"{msg['role']}: {msg['content']}\n"
        
        rewrite_prompt = f"""
        Erstelle basierend auf dem Chat-Verlauf eine präzise, eigenständige Suchanfrage.
        Ersetze Pronomen (er, sie, es, dort) durch die tatsächlichen Subjekte aus dem Verlauf.
        
        VERLAUF:
        {history_context}
        
        AKTUELLE FRAGE:
        {last_user_message}
        
        Antworte NUR mit der umformulierten Suchanfrage (keine Einleitung).
        """
        try:
            rewrite_res = llm.invoke([("user", rewrite_prompt)])
            search_query = rewrite_res.content.strip()
        except Exception as e:
            print(f"Fehler beim Query Rewriting: {e}")
            search_query = last_user_message

    # 3. HYBRID SUCHE (Mit Begrüßungs-Check)
    common_greetings = ["hi", "hallo", "hey", "moin", "servus", "guten tag"]
    if last_user_message.lower().strip() in common_greetings:
       context, graph = "Kein Kontext (Begrüßung)", "Keine Tripel (Begrüßung)"
    else:
       # Nur wenn es keine einfache Begrüßung ist, suchen wir in den Dokumenten
       context, graph = search_hybrid_graph(search_query)

    # 4. SYSTEM PROMPT
    system_prompt = f"""
    ### DEINE ROLLE ###
    Du bist der offizielle SCHNOOR Wissensexperte. Antworte basierend auf den bereitgestellten Daten.
    Dein Ziel: Maximale Vollständigkeit und Korrektheit.

    ### DEIN VERHALTEN ###
    1. BEGRÜẞUNG & SMALLTALK: Wenn der Nutzer dich grüßt oder allgemeine Fragen stellt (z.B. "Wie geht es dir?", "Wer bist du?"), antworte charmant und hilfsbereit mit deinem eigenen Wissen.
    2. FACHFRAGEN: Sobald eine Frage zu SCHNOOR, Projekten oder Fachthemen gestellt wird, priorisiere die DATENGRUNDLAGE.
    3. DATENTREUE: Fakten aus dem WISSENSGRAPH und TEXT-KONTEXT haben bei Fachfragen absolute Priorität.

    ### DATENGRUNDLAGE (Nur für Fachfragen) ###
    1. WISSENSGRAPH (Strukturierte Fakten):
    {graph}

    2. TEXT-KONTEXT (Detaillierte Belege):
    {context}

    ### ARBEITSANWEISUNG ###
    - Schritt 1: Entscheide, ob die Frage eine Begrüßung oder allgemeine Frage ist. Wenn ja, antworte direkt. Wenn Nein, nutze die Datengrundlage! 
    - Schritt 2: Nutze den WISSENSGRAPH als Master-Liste. Jedes Faktum dort MUSS in die Antwort.
    - Schritt 3: Ergänze Details aus dem TEXT-KONTEXT.
    - Schritt 4: Wenn Informationen fehlen, antworte: "Dazu liegen keine internen Dokumente vor."
    - Schritt 5: Verlinke am Ende JEDE genannte Quelle aus dem TEXT-KONTEXT.
    - Schritt 6: Falls KEINE internen Daten zu einer fachlichen Frage vorliegen: Antworte "Dazu liegen keine internen Dokumente vor, aber allgemein bekannt ist...",

    ### FORMATIERUNG ###
    - Nutze Markdown-Listen für Übersichtlichkeit.
    - Keine Sätze wie "Laut Dokument...". Antworte direkt.
    - ABSCHNITT QUELLEN: Liste alle verwendeten Quellen am Ende exakt so auf: [Titel](URL)
    """
    
    # 5. CHAT-HISTORIE FÜR DAS LLM AUFBEREITEN
    llm_messages = [("system", system_prompt)]
    
    if isinstance(messages, list):
        for msg in messages[:-1]:
            role = "user" if msg.get("role") == "user" else "assistant"
            llm_messages.append((role, msg.get("content", "")))

    llm_messages.append(("user", last_user_message))

    # --- DEBUG LOGGING ---
    print("\n" + "!"*60)
    print("🚀 FINALER PROMPT AN OLLAMA (QWEN-32B)")
    print("!"*60)
    
    for msg_type, content in llm_messages:
        print(f"\n--- ROLE: {msg_type.upper()} ---")
        print(content)
    
    print("\n" + "!"*60)
    print("ENDE DES PROMPTS - WARTE AUF GENERIERUNG...")
    print("!"*60 + "\n")

    # 6. ANTWORT GENERIEREN
    response = llm.invoke(llm_messages)
    
    return {
        "answer": response.content,
        "sources": str(context),
        "graph": str(graph)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)
