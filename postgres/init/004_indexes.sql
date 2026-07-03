CREATE INDEX IF NOT EXISTS idx_qa_raw_qa_items_ticker ON qa.raw_qa_items(ticker);
CREATE INDEX IF NOT EXISTS idx_qa_raw_qa_items_query_type ON qa.raw_qa_items(query_type);
CREATE INDEX IF NOT EXISTS idx_qa_bridge_candidates_is_bridge_candidate ON qa.bridge_candidates(is_bridge_candidate);
CREATE INDEX IF NOT EXISTS idx_qa_bridge_records_rewrite_type ON qa.bridge_records(rewrite_type);
CREATE INDEX IF NOT EXISTS idx_qa_bridge_records_rewrite_quality_status ON qa.bridge_records(rewrite_quality_status);
CREATE INDEX IF NOT EXISTS idx_qa_temporal_annotations_temporal_scope_type ON qa.temporal_annotations(temporal_scope_type);
CREATE INDEX IF NOT EXISTS idx_qa_m0_inputs_m0_input_setting ON qa.m0_inputs(m0_input_setting);
CREATE INDEX IF NOT EXISTS idx_qa_m0_predictions_method ON qa.m0_predictions(method);

CREATE INDEX IF NOT EXISTS idx_content_raw_documents_source_type ON content.raw_documents(source_type);
CREATE INDEX IF NOT EXISTS idx_content_documents_document_year ON content.documents(document_year);
CREATE INDEX IF NOT EXISTS idx_content_documents_document_year_source ON content.documents(document_year_source);
CREATE INDEX IF NOT EXISTS idx_content_documents_source_name ON content.documents(source_name);
CREATE INDEX IF NOT EXISTS idx_content_documents_metadata_quality_status ON content.documents(metadata_quality_status);
CREATE INDEX IF NOT EXISTS idx_content_document_entities_ticker ON content.document_entities(ticker);
CREATE INDEX IF NOT EXISTS idx_content_document_entities_entity_type ON content.document_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_content_document_entities_match_confidence ON content.document_entities(match_confidence);
CREATE INDEX IF NOT EXISTS idx_content_evidence_chunks_document_year ON content.evidence_chunks(document_year);
CREATE INDEX IF NOT EXISTS idx_content_evidence_chunks_doc_id ON content.evidence_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_content_evidence_chunks_source_name ON content.evidence_chunks(source_name);
CREATE INDEX IF NOT EXISTS idx_content_evidence_chunks_chunk_quality_status ON content.evidence_chunks(chunk_quality_status);
CREATE INDEX IF NOT EXISTS idx_content_evidence_chunks_mentioned_tickers ON content.evidence_chunks USING gin (mentioned_tickers);

DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_content_evidence_chunks_embedding_hnsw
    ON content.evidence_chunks
    USING hnsw (embedding vector_cosine_ops);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Skipping HNSW vector index; sequential vector scan still works: %', SQLERRM;
END $$;
