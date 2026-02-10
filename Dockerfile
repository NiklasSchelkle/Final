FROM python:3.11-slim

WORKDIR /app

# [cite_start]System-Abhängigkeiten erweitert für Docling & Grafik-Verarbeitung [cite: 1]
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Verzeichnisse für Modell-Caches UND Ingestion erstellen
# Wir legen /app/data/ingest explizit an, damit das Volume-Mapping sauber greift
RUN mkdir -p /app/models/flashrank /app/models/huggingface /app/data/ingest

COPY requirements.txt .
# [cite_start]Installiert alle Pakete inklusive docling, flashrank und langchain-Erweiterungen [cite: 2, 4]
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

# [cite_start]Kopiert den Code (api.py, engine.py, etc.) in den Container [cite: 3]
COPY . .

# [cite_start]Port für die API [cite: 3]
EXPOSE 8050

CMD ["python", "api.py"]