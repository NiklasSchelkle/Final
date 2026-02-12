SCHNOOR Hybrid RAG System 🚀

Dieses Repository enthält ein fortschrittliches Hybrid RAG (Retrieval-Augmented Generation) System, das Vektorsuche und Wissensgraphen kombiniert. Optimiert für den Betrieb auf NVIDIA GPU-Servern (z. B. L40S) unter Nutzung von Docker, Ollama und PostgreSQL.
📋 Inhaltsverzeichnis

    1. Voraussetzungen & Treiber

    2. Installation NVIDIA Toolkit

    3. Setup & Start

    4. Modell-Konfiguration

    5. Daten-Ingestion

    🔗 Zugriff

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

# GPG-Key für die Sicherheit hinzufügen
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Repository-Liste korrekt anlegen
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Toolkit installieren
 sudo apt-get update
 sudo apt-get install -y nvidia-container-toolkit

# Docker für GPU-Nutzung konfigurieren
 sudo nvidia-container-toolkit runtime configure --runtime=docker

# Docker neu starten, um Änderungen zu übernehmen
 sudo systemctl restart docker

Funktionstest (Hardware -> Treiber -> Docker)
Bash

# Prüfen, ob der Container die Grafikkarte sieht
 docker run --rm --gpus all nvidia/cuda:12.0.1-base-ubuntu22.04 nvidia-smi

3. Setup & Start
Repository klonen
Bash

 git clone https://github.com/NiklasSchelkle/Final.git
 cd Final

Konfiguration
Bash

# In der .env Datei Tokens und Pfade eintragen
 nano .env 

System starten
Bash

# Starten der Skripte und Images
 docker compose up -d

# Status prüfen
 docker ps -a

4. Modell-Konfiguration

Lade das LLM und das Embedding-Modell in den Ollama-Container.
Bash

# Embedding Model (auf Dimensionen in VektorErweiterung.sql achten)
 docker exec -it ollama ollama pull mxbai-embed-large 

# LLM
 docker exec -it ollama ollama pull mistral-nemo

5. Daten-Ingestion

Dokumente übertragen und in die Hybrid-Datenbank einlesen.
Ordner auf dem Server erstellen
Bash

 mkdir -p ~/Final/files_to_embed

Dokumente hochladen (vom lokalen PC)
PowerShell

# Befehl für Windows PowerShell
 scp -r "C:\Users\Dell\SchnoorfinalRAG\files_to_embed\*" root@86.38.238.152:~/Final/files_to_embed/

Einlesen & Embedden
Bash

# Startet das Ingestion-Skript im Backend-Container
docker exec -it rag_backend python ingestion.py

🔗 Zugriff

Frontend: https://schnoorki.knowladgebaseai.space/

Database Management: https://schnoordatabase.knowladgebaseai.space/
