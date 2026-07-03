CREATE TABLE IF NOT EXISTS content.raw_documents (
    doc_id TEXT PRIMARY KEY,
    source_type TEXT,
    raw_text TEXT NOT NULL,
    raw_json JSONB,
    text_length INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS content.documents (
    doc_id TEXT PRIMARY KEY REFERENCES content.raw_documents(doc_id),
    title TEXT,
    source_name TEXT,
    source_type TEXT,
    document_year INTEGER NULL,
    document_year_source TEXT,
    mentioned_years JSONB,
    candidate_years JSONB,
    title_extraction_status TEXT,
    year_extraction_status TEXT,
    metadata_quality_status TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS content.document_entities (
    doc_entity_id TEXT PRIMARY KEY,
    doc_id TEXT REFERENCES content.documents(doc_id),
    entity_text TEXT,
    ticker TEXT,
    entity_type TEXT,
    match_source TEXT,
    match_confidence TEXT,
    context_snippet TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS content.evidence_chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT REFERENCES content.documents(doc_id),
    chunk_index INTEGER,
    chunk_text TEXT NOT NULL,
    chunk_char_start INTEGER,
    chunk_char_end INTEGER,
    chunk_token_estimate INTEGER,
    document_year INTEGER NULL,
    document_year_source TEXT,
    mentioned_years JSONB,
    mentioned_tickers JSONB,
    source_name TEXT,
    source_type TEXT,
    title TEXT,
    chunk_quality_status TEXT,
    metadata JSONB,
    embedding VECTOR(8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS content.content_build_runs (
    run_id TEXT PRIMARY KEY,
    input_path TEXT,
    output_dir TEXT,
    total_lines INTEGER,
    valid_raw_documents INTEGER,
    skipped_lines INTEGER,
    normalized_documents INTEGER,
    documents_with_year INTEGER,
    documents_without_year INTEGER,
    documents_with_entity INTEGER,
    total_chunks INTEGER,
    chunks_with_year INTEGER,
    chunks_with_entity INTEGER,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS content.content_retrieval_runs (
    retrieval_run_id TEXT PRIMARY KEY,
    query_text TEXT,
    filter_year_start INTEGER NULL,
    filter_year_end INTEGER NULL,
    filter_ticker TEXT NULL,
    top_k INTEGER,
    retrieved_chunk_ids JSONB,
    retrieved_years JSONB,
    retrieved_tickers JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
