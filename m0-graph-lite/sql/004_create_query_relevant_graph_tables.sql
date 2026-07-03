CREATE TABLE IF NOT EXISTS m0_graph.query_relevant_chunks (
    m0_input_id TEXT,
    bridge_id TEXT,
    ticker TEXT,
    query_text TEXT,
    rewrite_type TEXT,
    year INTEGER,
    chunk_id TEXT,
    doc_id TEXT,
    chunk_text TEXT,
    relevance_score REAL,
    lexical_score REAL,
    ticker_match BOOLEAN,
    year_match BOOLEAN,
    rank_in_year INTEGER,
    match_reason TEXT,
    build_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (m0_input_id, year, chunk_id, build_version)
);

CREATE TABLE IF NOT EXISTS m0_graph.query_year_evidence (
    m0_input_id TEXT,
    bridge_id TEXT,
    ticker TEXT,
    query_text TEXT,
    rewrite_type TEXT,
    year INTEGER,
    total_candidate_chunks INTEGER,
    query_relevant_chunk_count INTEGER,
    max_chunk_relevance REAL,
    avg_top_chunk_relevance REAL,
    sum_top_chunk_relevance REAL,
    year_relevance_score REAL,
    year_has_relevant_evidence BOOLEAN,
    top_chunk_ids JSONB,
    top_chunk_scores JSONB,
    build_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (m0_input_id, year, build_version)
);

CREATE TABLE IF NOT EXISTS m0_graph.candidate_query_graph_features (
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
    query_relevant_year_count INTEGER,
    query_relevant_year_ratio REAL,
    window_relevance_mean REAL,
    window_relevance_max REAL,
    window_relevance_min REAL,
    window_relevance_sum REAL,
    window_relevance_balance REAL,
    semantic_gap_count INTEGER,
    max_semantic_gap_length INTEGER,
    prefix_query_support REAL,
    suffix_query_support REAL,
    normalized_query_relevance_center REAL,
    directional_relevance_fit REAL,
    build_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (candidate_window_id, build_version)
);

CREATE INDEX IF NOT EXISTS idx_query_relevant_chunks_input
ON m0_graph.query_relevant_chunks(m0_input_id, build_version);

CREATE INDEX IF NOT EXISTS idx_query_relevant_chunks_ticker_year
ON m0_graph.query_relevant_chunks(ticker, year, build_version);

CREATE INDEX IF NOT EXISTS idx_query_year_evidence_input
ON m0_graph.query_year_evidence(m0_input_id, build_version);

CREATE INDEX IF NOT EXISTS idx_candidate_query_graph_features_input
ON m0_graph.candidate_query_graph_features(m0_input_id, build_version);

DROP VIEW IF EXISTS m0_graph.v_candidate_query_graph_feature_summary;
DROP VIEW IF EXISTS m0_graph.v_query_year_evidence_summary;

CREATE OR REPLACE VIEW m0_graph.v_query_year_evidence_summary AS
SELECT
    rewrite_type,
    build_version,
    COUNT(*) AS count,
    AVG(query_relevant_chunk_count) AS avg_query_relevant_chunk_count,
    AVG(year_relevance_score) AS avg_year_relevance_score,
    AVG(CASE WHEN year_has_relevant_evidence THEN 1.0 ELSE 0.0 END) AS relevant_year_rate,
    AVG(max_chunk_relevance) AS avg_max_chunk_relevance
FROM m0_graph.query_year_evidence
GROUP BY rewrite_type, build_version;

CREATE OR REPLACE VIEW m0_graph.v_candidate_query_graph_feature_summary AS
SELECT
    rewrite_type,
    window_type,
    build_version,
    COUNT(*) AS count,
    AVG(query_relevant_year_ratio) AS avg_query_relevant_year_ratio,
    AVG(window_relevance_mean) AS avg_window_relevance_mean,
    AVG(window_relevance_balance) AS avg_window_relevance_balance,
    AVG(semantic_gap_count) AS avg_semantic_gap_count,
    AVG(prefix_query_support) AS avg_prefix_query_support,
    AVG(suffix_query_support) AS avg_suffix_query_support,
    AVG(directional_relevance_fit) AS avg_directional_relevance_fit
FROM m0_graph.candidate_query_graph_features
GROUP BY rewrite_type, window_type, build_version;
