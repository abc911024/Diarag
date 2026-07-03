CREATE TABLE IF NOT EXISTS m0_graph.chunk_temporal_qualifications (
    m0_input_id TEXT,
    bridge_id TEXT,
    ticker TEXT,
    query_text TEXT,
    rewrite_type TEXT,
    year INTEGER,
    chunk_id TEXT,
    doc_id TEXT,
    source_build_version TEXT,
    qualification_build_version TEXT,
    entity_match BOOLEAN,
    text_temporal_anchor BOOLEAN,
    metadata_temporal_anchor BOOLEAN,
    temporal_anchor_strength TEXT,
    temporal_anchor BOOLEAN,
    temporal_signal BOOLEAN,
    signal_types JSONB,
    boundary_value TEXT,
    evidence_label TEXT,
    qualification_label TEXT,
    use_for_inside_relevance BOOLEAN,
    use_for_evidence_coverage BOOLEAN,
    use_for_boundary_contrast BOOLEAN,
    use_for_trend_coherence BOOLEAN,
    confidence REAL,
    extracted_evidence JSONB,
    short_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (m0_input_id, year, chunk_id, source_build_version, qualification_build_version)
);

CREATE INDEX IF NOT EXISTS idx_chunk_temporal_qualifications_input
ON m0_graph.chunk_temporal_qualifications(m0_input_id, source_build_version, qualification_build_version);

CREATE INDEX IF NOT EXISTS idx_chunk_temporal_qualifications_label
ON m0_graph.chunk_temporal_qualifications(qualification_label, qualification_build_version);

ALTER TABLE m0_graph.chunk_temporal_qualifications
ADD COLUMN IF NOT EXISTS text_temporal_anchor BOOLEAN,
ADD COLUMN IF NOT EXISTS metadata_temporal_anchor BOOLEAN,
ADD COLUMN IF NOT EXISTS temporal_anchor_strength TEXT,
ADD COLUMN IF NOT EXISTS evidence_label TEXT,
ADD COLUMN IF NOT EXISTS use_for_inside_relevance BOOLEAN,
ADD COLUMN IF NOT EXISTS use_for_evidence_coverage BOOLEAN,
ADD COLUMN IF NOT EXISTS use_for_boundary_contrast BOOLEAN,
ADD COLUMN IF NOT EXISTS use_for_trend_coherence BOOLEAN;

CREATE TABLE IF NOT EXISTS m0_graph.e7_candidate_boundary_diagnostics (
    candidate_window_id TEXT,
    m0_input_id TEXT,
    bridge_id TEXT,
    ticker TEXT,
    query_text TEXT,
    rewrite_type TEXT,
    start_year INTEGER,
    end_year INTEGER,
    window_type TEXT,
    window_length INTEGER,
    raw_chunk_count INTEGER,
    inside_relevance_chunk_count INTEGER,
    evidence_coverage_chunk_count INTEGER,
    boundary_contrast_chunk_count INTEGER,
    trend_coherence_chunk_count INTEGER,
    qualified_chunk_count INTEGER,
    boundary_temporal_evidence_count INTEGER,
    general_temporal_evidence_count INTEGER,
    supporting_context_count INTEGER,
    low_value_noise_count INTEGER,
    qualified_evidence_year_count INTEGER,
    boundary_evidence_year_count INTEGER,
    qualified_evidence_year_coverage REAL,
    boundary_evidence_year_coverage REAL,
    background_noise_ratio REAL,
    source_build_version TEXT,
    qualification_build_version TEXT,
    boundary_build_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (candidate_window_id, boundary_build_version)
);

ALTER TABLE m0_graph.e7_candidate_boundary_diagnostics
ADD COLUMN IF NOT EXISTS inside_relevance_chunk_count INTEGER,
ADD COLUMN IF NOT EXISTS evidence_coverage_chunk_count INTEGER,
ADD COLUMN IF NOT EXISTS boundary_contrast_chunk_count INTEGER,
ADD COLUMN IF NOT EXISTS trend_coherence_chunk_count INTEGER,
ADD COLUMN IF NOT EXISTS boundary_temporal_evidence_count INTEGER,
ADD COLUMN IF NOT EXISTS general_temporal_evidence_count INTEGER,
ADD COLUMN IF NOT EXISTS boundary_evidence_year_count INTEGER,
ADD COLUMN IF NOT EXISTS boundary_evidence_year_coverage REAL;

DROP VIEW IF EXISTS m0_graph.v_e7_chunk_qualification_summary;
DROP VIEW IF EXISTS m0_graph.v_e7_candidate_boundary_diagnostic_summary;

CREATE OR REPLACE VIEW m0_graph.v_e7_chunk_qualification_summary AS
SELECT
    rewrite_type,
    qualification_build_version,
    COUNT(*) AS chunk_count,
    AVG(CASE WHEN entity_match THEN 1.0 ELSE 0.0 END) AS entity_match_rate,
    AVG(CASE WHEN temporal_anchor THEN 1.0 ELSE 0.0 END) AS temporal_anchor_rate,
    AVG(CASE WHEN temporal_signal THEN 1.0 ELSE 0.0 END) AS temporal_signal_rate,
    AVG(CASE WHEN qualification_label = 'qualified_temporal_evidence' THEN 1.0 ELSE 0.0 END) AS qualified_rate,
    AVG(CASE WHEN evidence_label = 'boundary_temporal_evidence' THEN 1.0 ELSE 0.0 END) AS boundary_temporal_evidence_rate,
    AVG(CASE WHEN evidence_label = 'general_temporal_evidence' THEN 1.0 ELSE 0.0 END) AS general_temporal_evidence_rate,
    AVG(CASE WHEN qualification_label = 'supporting_context' THEN 1.0 ELSE 0.0 END) AS supporting_context_rate,
    AVG(CASE WHEN qualification_label = 'low_value_noise' THEN 1.0 ELSE 0.0 END) AS low_value_noise_rate,
    AVG(CASE WHEN use_for_inside_relevance THEN 1.0 ELSE 0.0 END) AS inside_relevance_usage_rate,
    AVG(CASE WHEN use_for_evidence_coverage THEN 1.0 ELSE 0.0 END) AS evidence_coverage_usage_rate,
    AVG(CASE WHEN use_for_boundary_contrast THEN 1.0 ELSE 0.0 END) AS boundary_contrast_usage_rate,
    AVG(CASE WHEN use_for_trend_coherence THEN 1.0 ELSE 0.0 END) AS trend_coherence_usage_rate,
    AVG(confidence) AS avg_confidence
FROM m0_graph.chunk_temporal_qualifications
GROUP BY rewrite_type, qualification_build_version;

CREATE OR REPLACE VIEW m0_graph.v_e7_candidate_boundary_diagnostic_summary AS
SELECT
    rewrite_type,
    window_type,
    boundary_build_version,
    COUNT(*) AS count,
    AVG(qualified_evidence_year_coverage) AS avg_qualified_evidence_year_coverage,
    AVG(boundary_evidence_year_coverage) AS avg_boundary_evidence_year_coverage,
    AVG(background_noise_ratio) AS avg_background_noise_ratio,
    AVG(inside_relevance_chunk_count) AS avg_inside_relevance_chunk_count,
    AVG(evidence_coverage_chunk_count) AS avg_evidence_coverage_chunk_count,
    AVG(boundary_contrast_chunk_count) AS avg_boundary_contrast_chunk_count,
    AVG(trend_coherence_chunk_count) AS avg_trend_coherence_chunk_count,
    AVG(qualified_chunk_count) AS avg_qualified_chunk_count,
    AVG(boundary_temporal_evidence_count) AS avg_boundary_temporal_evidence_count,
    AVG(general_temporal_evidence_count) AS avg_general_temporal_evidence_count,
    AVG(low_value_noise_count) AS avg_low_value_noise_count
FROM m0_graph.e7_candidate_boundary_diagnostics
GROUP BY rewrite_type, window_type, boundary_build_version;
