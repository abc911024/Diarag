CREATE TABLE IF NOT EXISTS m0_graph.year_trend_signals (
    m0_input_id TEXT,
    bridge_id TEXT,
    ticker TEXT,
    query_text TEXT,
    rewrite_type TEXT,
    year INTEGER,
    hybrid_year_relevance_score REAL,
    year_has_hybrid_relevant_evidence BOOLEAN,
    trend_signal_score REAL,
    up_signal_score REAL,
    down_signal_score REAL,
    stable_signal_score REAL,
    volatile_signal_score REAL,
    dominant_trend_direction TEXT,
    top_trend_terms JSONB,
    top_chunk_ids JSONB,
    build_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (m0_input_id, year, build_version)
);

CREATE TABLE IF NOT EXISTS m0_graph.candidate_boundary_features (
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
    timeline_min_year INTEGER,
    timeline_max_year INTEGER,
    timeline_length INTEGER,
    inside_relevance_mean REAL,
    inside_relevance_std REAL,
    inside_coherence REAL,
    left_context_relevance REAL,
    right_context_relevance REAL,
    left_boundary_contrast REAL,
    right_boundary_contrast REAL,
    boundary_contrast REAL,
    trend_signal_mean REAL,
    trend_coherence REAL,
    dominant_window_trend_direction TEXT,
    intent_fit REAL,
    length_fit REAL,
    boundary_score_raw REAL,
    context_size INTEGER,
    build_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (candidate_window_id, build_version)
);

CREATE INDEX IF NOT EXISTS idx_year_trend_signals_input
ON m0_graph.year_trend_signals(m0_input_id, build_version);

CREATE INDEX IF NOT EXISTS idx_candidate_boundary_features_input
ON m0_graph.candidate_boundary_features(m0_input_id, build_version);

DROP VIEW IF EXISTS m0_graph.v_candidate_boundary_feature_summary;
DROP VIEW IF EXISTS m0_graph.v_year_trend_signal_summary;

CREATE OR REPLACE VIEW m0_graph.v_year_trend_signal_summary AS
SELECT
    rewrite_type,
    build_version,
    COUNT(*) AS count,
    AVG(hybrid_year_relevance_score) AS avg_hybrid_year_relevance_score,
    AVG(trend_signal_score) AS avg_trend_signal_score,
    AVG(up_signal_score) AS avg_up_signal_score,
    AVG(down_signal_score) AS avg_down_signal_score,
    AVG(stable_signal_score) AS avg_stable_signal_score,
    AVG(volatile_signal_score) AS avg_volatile_signal_score
FROM m0_graph.year_trend_signals
GROUP BY rewrite_type, build_version;

CREATE OR REPLACE VIEW m0_graph.v_candidate_boundary_feature_summary AS
SELECT
    rewrite_type,
    window_type,
    build_version,
    COUNT(*) AS count,
    AVG(inside_relevance_mean) AS avg_inside_relevance_mean,
    AVG(inside_coherence) AS avg_inside_coherence,
    AVG(boundary_contrast) AS avg_boundary_contrast,
    AVG(trend_signal_mean) AS avg_trend_signal_mean,
    AVG(trend_coherence) AS avg_trend_coherence,
    AVG(intent_fit) AS avg_intent_fit,
    AVG(length_fit) AS avg_length_fit,
    AVG(boundary_score_raw) AS avg_boundary_score_raw
FROM m0_graph.candidate_boundary_features
GROUP BY rewrite_type, window_type, build_version;
