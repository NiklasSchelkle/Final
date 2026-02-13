-- 1. Erweiterungen aktivieren
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Tabellen für das hierarchische RAG (Vektorsuche)
CREATE TABLE IF NOT EXISTS parent_documents (
    id UUID PRIMARY KEY,
    title TEXT,
    full_text TEXT,
    metadata JSONB,
    source_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP -- NEU: Zeitstempel für Aktualität
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    parent_id UUID REFERENCES parent_documents(id) ON DELETE CASCADE,
    content TEXT,
    embedding vector(1024) 
);

-- 3. Tabellen für den Knowledge Graph
CREATE TABLE IF NOT EXISTS document_nodes (
    id UUID PRIMARY KEY,
    entity_name TEXT UNIQUE,
    entity_type TEXT
);

CREATE TABLE IF NOT EXISTS document_edges (
    id UUID PRIMARY KEY,
    source_node_id UUID REFERENCES document_nodes(id) ON DELETE CASCADE,
    target_node_id UUID REFERENCES document_nodes(id) ON DELETE CASCADE,
    relation_type TEXT,
    source_doc_id UUID REFERENCES parent_documents(id) ON DELETE CASCADE,
    confidence SMALLINT DEFAULT 5 -- NEU: Vertrauenswürdigkeit des Fakts
);

-- 4. Performance-Optimierung (Indizes)
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_edges_source ON document_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON document_edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON document_nodes(entity_name);
CREATE INDEX IF NOT EXISTS idx_edges_confidence ON document_edges(confidence); -- NEU: Index für schnelles Filtern nach Qualität

