CREATE OR REPLACE VIEW qa.v_bridge_dataset_overview AS
SELECT
    br.bridge_id,
    br.source_id,
    br.rewritten_query,
    br.rewrite_type,
    br.rewrite_quality_status,
    br.ticker,
    br.stock_name,
    ta.gold_range,
    ta.acceptable_ranges,
    ta.temporal_scope_type
FROM qa.bridge_records br
JOIN qa.temporal_annotations ta ON ta.bridge_id = br.bridge_id;

CREATE OR REPLACE VIEW qa.v_m0_inputs_with_targets AS
SELECT
    mi.m0_input_id,
    mi.bridge_id,
    mi.m0_input_setting,
    br.rewritten_query,
    br.rewrite_type,
    mi.m0_input_json,
    ta.gold_range,
    ta.acceptable_ranges,
    ta.temporal_scope_type
FROM qa.m0_inputs mi
JOIN qa.bridge_records br ON br.bridge_id = mi.bridge_id
JOIN qa.temporal_annotations ta ON ta.bridge_id = mi.bridge_id;

CREATE OR REPLACE VIEW qa.v_m0_metrics_summary AS
SELECT
    mp.method,
    mi.m0_input_setting,
    br.rewrite_type,
    COUNT(*) AS count,
    AVG(mm.overlap_score) AS avg_overlap_score,
    AVG(mm.mean_boundary_error) AS avg_mean_boundary_error,
    AVG(CASE WHEN mm.acceptable_accuracy THEN 1.0 ELSE 0.0 END) AS acceptable_accuracy_rate,
    AVG(CASE WHEN mm.exact_match_accuracy THEN 1.0 ELSE 0.0 END) AS exact_match_rate
FROM qa.m0_predictions mp
JOIN qa.m0_metrics mm ON mm.prediction_id = mp.prediction_id
JOIN qa.m0_inputs mi ON mi.m0_input_id = mp.m0_input_id
JOIN qa.bridge_records br ON br.bridge_id = mi.bridge_id
GROUP BY mp.method, mi.m0_input_setting, br.rewrite_type;

CREATE OR REPLACE VIEW content.v_content_document_overview AS
SELECT
    doc_id,
    title,
    source_name,
    document_year,
    document_year_source,
    source_type,
    mentioned_years,
    metadata_quality_status
FROM content.documents;

CREATE OR REPLACE VIEW content.v_content_entity_year_coverage AS
SELECT
    de.ticker,
    ec.document_year,
    COUNT(DISTINCT de.doc_id) AS doc_count,
    COUNT(DISTINCT ec.chunk_id) AS chunk_count
FROM content.document_entities de
LEFT JOIN content.evidence_chunks ec
    ON ec.doc_id = de.doc_id
    AND (de.ticker IS NULL OR ec.mentioned_tickers @> to_jsonb(ARRAY[de.ticker]))
WHERE de.ticker IS NOT NULL
GROUP BY de.ticker, ec.document_year;

CREATE OR REPLACE VIEW content.v_content_chunk_overview AS
SELECT
    chunk_id,
    doc_id,
    title,
    document_year,
    mentioned_tickers,
    source_name,
    chunk_quality_status
FROM content.evidence_chunks;

CREATE OR REPLACE VIEW content.v_content_quality_summary AS
SELECT
    (SELECT COUNT(*) FROM content.documents) AS total_documents,
    (SELECT COUNT(*) FROM content.documents WHERE document_year IS NOT NULL) AS documents_with_year,
    (SELECT COUNT(*) FROM content.documents WHERE document_year IS NULL) AS documents_without_year,
    (SELECT COUNT(*) FROM content.evidence_chunks) AS total_chunks,
    (SELECT COUNT(*) FROM content.evidence_chunks WHERE document_year IS NOT NULL) AS chunks_with_year,
    (SELECT COUNT(*) FROM content.evidence_chunks WHERE jsonb_array_length(COALESCE(mentioned_tickers, '[]'::jsonb)) > 0) AS chunks_with_ticker;
