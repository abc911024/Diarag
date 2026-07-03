CREATE TABLE IF NOT EXISTS m0_graph.hybrid_query_relevant_chunks (
    m0_input_id TEXT,
    bridge_id TEXT,
    ticker TEXT,
    query_text TEXT,
    rewrite_type TEXT,
    year INTEGER,
    chunk_id TEXT,
    doc_id TEXT,
    chunk_text TEXT,
    lexical_score REAL,
    dense_similarity_score REAL,
    signal_bonus REAL,
    hybrid_relevance_score REAL,
    ticker_match BOOLEAN,
    market_signal_match BOOLEAN,
    numeric_signal_match BOOLEAN,
    year_match BOOLEAN,
    rank_in_year INTEGER,
    match_reason TEXT,
    dense_model_name TEXT,
    build_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (m0_input_id, year, chunk_id, build_version)
);

CREATE TABLE IF NOT EXISTS m0_graph.hybrid_query_year_evidence (
    m0_input_id TEXT,
    bridge_id TEXT,
    ticker TEXT,
    query_text TEXT,
    rewrite_type TEXT,
    year INTEGER,
    total_candidate_chunks INTEGER,
    hybrid_relevant_chunk_count INTEGER,
    max_hybrid_relevance REAL,
    avg_top_hybrid_relevance REAL,
    sum_top_hybrid_relevance REAL,
    avg_dense_similarity REAL,
    avg_lexical_score REAL,
    hybrid_year_relevance_score REAL,
    year_has_hybrid_relevant_evidence BOOLEAN,
    top_chunk_ids JSONB,
    top_chunk_scores JSONB,
    dense_model_name TEXT,
    build_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (m0_input_id, year, build_version)
);

CREATE TABLE IF NOT EXISTS m0_graph.hybrid_candidate_query_graph_features (
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
    hybrid_relevant_year_count INTEGER,
    hybrid_relevant_year_ratio REAL,
    hybrid_window_relevance_mean REAL,
    hybrid_window_relevance_max REAL,
    hybrid_window_relevance_min REAL,
    hybrid_window_relevance_sum REAL,
    hybrid_window_relevance_balance REAL,
    hybrid_semantic_gap_count INTEGER,
    max_hybrid_semantic_gap_length INTEGER,
    prefix_hybrid_query_support REAL,
    suffix_hybrid_query_support REAL,
    normalized_hybrid_relevance_center REAL,
    hybrid_directional_relevance_fit REAL,
    dense_model_name TEXT,
    build_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (candidate_window_id, build_version)
);

CREATE INDEX IF NOT EXISTS idx_hybrid_query_relevant_chunks_input
ON m0_graph.hybrid_query_relevant_chunks(m0_input_id, build_version);

CREATE INDEX IF NOT EXISTS idx_hybrid_query_relevant_chunks_ticker_year
ON m0_graph.hybrid_query_relevant_chunks(ticker, year, build_version);

CREATE INDEX IF NOT EXISTS idx_hybrid_query_year_evidence_input
ON m0_graph.hybrid_query_year_evidence(m0_input_id, build_version);

CREATE INDEX IF NOT EXISTS idx_hybrid_candidate_query_graph_features_input
ON m0_graph.hybrid_candidate_query_graph_features(m0_input_id, build_version);

DROP VIEW IF EXISTS m0_graph.v_hybrid_candidate_query_graph_feature_summary;
DROP VIEW IF EXISTS m0_graph.v_hybrid_query_year_evidence_summary;

CREATE OR REPLACE VIEW m0_graph.v_hybrid_query_year_evidence_summary AS
SELECT
    rewrite_type,
    build_version,
    COUNT(*) AS count,
    AVG(hybrid_relevant_chunk_count) AS avg_hybrid_relevant_chunk_count,
    AVG(hybrid_year_relevance_score) AS avg_hybrid_year_relevance_score,
    AVG(CASE WHEN year_has_hybrid_relevant_evidence THEN 1.0 ELSE 0.0 END) AS hybrid_relevant_year_rate,
    AVG(max_hybrid_relevance) AS avg_max_hybrid_relevance,
    AVG(avg_dense_similarity) AS avg_dense_similarity,
    AVG(avg_lexical_score) AS avg_lexical_score
FROM m0_graph.hybrid_query_year_evidence
GROUP BY rewrite_type, build_version;

CREATE OR REPLACE VIEW m0_graph.v_hybrid_candidate_query_graph_feature_summary AS
SELECT
    rewrite_type,
    window_type,
    build_version,
    COUNT(*) AS count,
    AVG(hybrid_relevant_year_ratio) AS avg_hybrid_relevant_year_ratio,
    AVG(hybrid_window_relevance_mean) AS avg_hybrid_window_relevance_mean,
    AVG(hybrid_window_relevance_balance) AS avg_hybrid_window_relevance_balance,
    AVG(hybrid_semantic_gap_count) AS avg_hybrid_semantic_gap_count,
    AVG(prefix_hybrid_query_support) AS avg_prefix_hybrid_query_support,
    AVG(suffix_hybrid_query_support) AS avg_suffix_hybrid_query_support,
    AVG(hybrid_directional_relevance_fit) AS avg_hybrid_directional_relevance_fit
FROM m0_graph.hybrid_candidate_query_graph_features
GROUP BY rewrite_type, window_type, build_version;
