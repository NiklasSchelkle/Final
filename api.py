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
    # Wir extrahieren die letzte Nachricht und den Verlauf
    if isinstance(query.question, list):
        messages = query.question
        last_user_message = messages[-1]["content"] if messages else ""
    else:
        messages = []
        last_user_message = query.question

    # 2. QUERY REWRITING (Das "Such-Gedächtnis")
    # Wir erstellen eine Suchanfrage, die Pronomen durch echte Namen ersetzt
    search_query = last_user_message
    if len(messages) > 1:
        # Wir fassen die letzten 3 Nachrichten zusammen, um Kontext zu gewinnen
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
            # Wir nutzen das LLM, um die Frage für die Vektorsuche zu optimieren
            rewrite_res = llm.invoke([("user", rewrite_prompt)])
            search_query = rewrite_res.content.strip()
        except Exception as e:
            print(f"Fehler beim Query Rewriting: {e}")
            search_query = last_user_message

    # 3. HYBRID SUCHE (mit der optimierten Suchanfrage)
    context, graph = search_hybrid_graph(search_query)
    
    # 4. SYSTEM PROMPT (Deine bewährten Anweisungen)
    system_prompt = f"""
    ### DEINE ROLLE ###
    Du bist der offizielle SCHNOOR Wissensexperte. Antworte basierend auf den bereitgestellten Daten.
    Dein Ziel: Maximale Vollständigkeit und Korrektheit.

    ### DATENGRUNDLAGE ###
    1. WISSENSGRAPH (Strukturierte Fakten):
    {graph}

    2. TEXT-KONTEXT (Detaillierte Belege):
    {context}

    ### ARBEITSANWEISUNG ###
    - Schritt 1: Nutze den WISSENSGRAPH als Master-Liste. Jedes Faktum dort MUSS in die Antwort.
    - Schritt 2: Ergänze Details aus dem TEXT-KONTEXT.
    - Schritt 3: Wenn Informationen fehlen, antworte: "Dazu liegen keine internen Dokumente vor."
    - Schritt 4: Verlinke am Ende JEDE genannte Quelle aus dem TEXT-KONTEXT.

    ### FORMATIERUNG ###
    - Nutze Markdown-Listen für Übersichtlichkeit.
    - Keine Sätze wie "Laut Dokument...". Antworte direkt.
    - ABSCHNITT QUELLEN: Liste alle verwendeten Quellen am Ende exakt so auf: [Titel](URL)
    """
    
    # 5. CHAT-HISTORIE FÜR DAS LLM AUFBEREITEN
    llm_messages = [("system", system_prompt)]
    
    if isinstance(messages, list):
        # Alle Nachrichten außer der letzten (die kommt als 'user' prompt)
        for msg in messages[:-1]:
            role = "user" if msg.get("role") == "user" else "assistant"
            llm_messages.append((role, msg.get("content", "")))

    # Die aktuelle Frage hinzufügen
    llm_messages.append(("user", last_user_message))

    # --- NEU: DEBUG LOGGING FÜR DEN FINALEN PROMPT ---
    print("\n" + "!"*60)
    print("🚀 FINALER PROMPT AN OLLAMA (MISTRAL-NEMO)")
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
