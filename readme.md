SCHNOOR Hybrid RAG System 🚀

Dieses Repository enthält ein fortschrittliches Hybrid RAG (Retrieval-Augmented Generation) System, das Vektorsuche und Wissensgraphen kombiniert. Optimiert für den Betrieb auf NVIDIA GPU-Servern (z.B. L40S) unter Nutzung von Docker, Ollama und PostgreSQL.
📋 Inhaltsverzeichnis

    Voraussetzungen & Treiber

    Installation NVIDIA Toolkit

    Setup & Start

    Modell-Konfiguration

    Daten-Ingestion

1. Voraussetzungen & Treiber

Bevor das System gestartet werden kann, müssen Docker und die NVIDIA-Treiber auf dem Host-System installiert sein.
Docker installieren
Bash

# Offizielles Docker-Installationsskript laden und ausführen
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

NVIDIA-Treiber installieren
Bash

# Paketlisten aktualisieren
sudo apt-get update

# Empfohlenen stabilen Server-Treiber installieren
sudo apt-get install -y nvidia-driver-550-server nvidia-utils-550-server

# WICHTIG: Server neu starten, um Treiber zu laden
sudo reboot

2. Installation NVIDIA Toolkit

Die Brücke zwischen Docker und der GPU.
Bash

# GPG-Key hinzufügen
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Repository-Liste anlegen
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Toolkit installieren
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker konfigurieren und neu starten
sudo nvidia-container-toolkit runtime configure --runtime=docker
sudo systemctl restart docker

Funktionstest

Überprüfe, ob Docker Zugriff auf die GPU hat:
Bash

docker run --rm --gpus all nvidia/cuda:12.0.1-base-ubuntu22.04 nvidia-smi

3. Setup & Start
Repository klonen
Bash

git clone https://github.com/NiklasSchelkle/Final.git
cd Final

Konfiguration

Erstelle eine .env Datei und trage deine Tokens und Pfade ein:
Bash

nano .env

System starten
Bash

# Images bauen und Container im Hintergrund starten
docker compose up -d

# Status der Container prüfen
docker ps -a

4. Modell-Konfiguration

Lade die benötigten Modelle in den Ollama-Container.
Bash

# Embedding Model (mxbai-embed-large)
docker exec -it ollama ollama pull mxbai-embed-large

# LLM (Mistral-Nemo)
docker exec -it ollama ollama pull mistral-nemo

5. Daten-Ingestion

Um Dokumente (PDFs, Docs, etc.) in das System einzuspielen:
Dokumente übertragen

Erstelle den Zielordner und lade deine Dateien von deinem lokalen Rechner hoch:
Bash

# Auf dem Server:
mkdir -p ~/Final/files_to_embed

# Auf deinem Laptop (Beispiel Pfad):
scp -r "C:\Users\Dell\SchnoorfinalRAG\files_to_embed\*" root@86.38.238.152:~/Final/files_to_embed/

Einlesen & Embedden

Starte das Ingestion-Skript, um Vektoren und Graph-Tripel zu erzeugen:
Bash

docker exec -it rag_backend python ingestion.py

🔗 Zugriff

    Frontend: https://schnoorki.knowladgebaseai.space/

    Database Management: https://schnoordatabase.knowladgebaseai.space/
