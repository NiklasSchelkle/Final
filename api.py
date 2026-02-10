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
        Erstelle basierend auf dem Chat-Verlauf eine präzise, eigenständige Suchanfrage für eine Datenbank.
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
    Du bist ein professioneller KI-Assistent für das Unternehmen SCHNOOR. 
    Deine Aufgabe ist es, eine präzise, zusammenhängende Antwort basierend auf internen Fakten zu liefern. Deine Aufgabe ist also die VOLLSTÄNDIGE Wiedergabe von Fakten.

    ### DATENGRUNDLAGE ###
    WISSENSGRAPH (Strukturierte Fakten): {graph}
    TEXT-KONTEXT (Ausführliche Belege): {context}

    ### DEINE ANWEISUNGEN ###
    1. Nutze den Chatverlauf um den Context zu verstehen.
    2. Antworte direkt und strukturiert. Erstelle EINE einzige, vollständige Liste oder Zusammenfassung.
    3. Nutze den WISSENSGRAPH als Master-Liste für die Vollständigkeit.Alles was dort steht, muss in die Antwort.
    4. Nutze den TEXT-KONTEXT für Details. Fehlt der Text zu einem Graph-Eintrag, nenne ihn trotzdem kurz.
    5. Wenn Info fehlt: "Dazu habe ich keine internen Dokumente, aber allgemein bekannt ist...".
    
    ### FORMATIERUNG ###
    - Keine Meta-Diskussionen.
    - Quellen am Ende übersichtlich auflisten: [Titel](URL).
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
    
    # 6. ANTWORT GENERIEREN
    response = llm.invoke(llm_messages)
    
    return {
        "answer": response.content,
        "sources": str(context),
        "graph": str(graph)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)