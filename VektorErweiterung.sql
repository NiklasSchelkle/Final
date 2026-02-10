-- 1. Erweiterungen aktivieren
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Tabellen für das hierarchische RAG (Vektorsuche)
-- Speichert große Textblöcke für den LLM-Kontext
CREATE TABLE IF NOT EXISTS parent_documents (
    id UUID PRIMARY KEY,
    title TEXT,
    full_text TEXT,
    metadata JSONB,
    source_url TEXT
);

-- Speichert kleine Schnipsel und deren Vektoren für die Ähnlichkeitssuche
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    parent_id UUID REFERENCES parent_documents(id) ON DELETE CASCADE,
    content TEXT,
    embedding vector(1024) -- für OpenAI 'text-embedding-3-small' --> 1536,  für mxbai-embed-large 1024
);

-- 3. Tabellen für den Knowledge Graph (Beziehungs-Suche)
-- Speichert eindeutige Entitäten (Personen, Orte, Projekte, etc.)
CREATE TABLE IF NOT EXISTS document_nodes (
    id UUID PRIMARY KEY,
    entity_name TEXT UNIQUE, -- UNIQUE verhindert doppelte Begriffe
    entity_type TEXT
);

-- Speichert die Beziehungen zwischen den Entitäten
CREATE TABLE IF NOT EXISTS document_edges (
    id UUID PRIMARY KEY,
    source_node_id UUID REFERENCES document_nodes(id) ON DELETE CASCADE,
    target_node_id UUID REFERENCES document_nodes(id) ON DELETE CASCADE,
    relation_type TEXT,
    source_doc_id UUID REFERENCES parent_documents(id) ON DELETE CASCADE
);

-- 4. Performance-Optimierung (Indizes)
-- HNSW Index für blitzschnelle Vektorsuche bei tausenden Dokumenten
CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Indizes für schnelle Graph-Abfragen (Traversierung)
CREATE INDEX idx_edges_source ON document_edges(source_node_id);
CREATE INDEX idx_edges_target ON document_edges(target_node_id);
CREATE INDEX idx_nodes_name ON document_nodes(entity_name);
