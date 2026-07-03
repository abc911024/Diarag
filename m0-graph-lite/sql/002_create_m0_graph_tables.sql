CREATE TABLE IF NOT EXISTS m0_graph.ticker_year_evidence (
    ticker TEXT,
    year INTEGER,
    doc_count INTEGER,
    chunk_count INTEGER,
    has_evidence BOOLEAN,
    density_score REAL,
    source_schema TEXT DEFAULT 'content',
    build_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, year, build_version)
);

CREATE TABLE IF NOT EXISTS m0_graph.candidate_year_coverage (
    candidate_window_id TEXT,
    m0_input_id TEXT,
    bridge_id TEXT,
    ticker TEXT,
    year INTEGER,
    start_year INTEGER,
    end_year INTEGER,
    window_type TEXT,
    has_evidence BOOLEAN,
    chunk_count INTEGER,
    doc_count INTEGER,
    is_gap BOOLEAN,
    build_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (candidate_window_id, year, build_version)
);

CREATE TABLE IF NOT EXISTS m0_graph.candidate_graph_features (
    candidate_window_id TEXT,
    m0_input_id TEXT,
    bridge_id TEXT,
    ticker TEXT,
    start_year INTEGER,
    end_year INTEGER,
    window_type TEXT,
    window_length INTEGER,
    timeline_min_year INTEGER,
    timeline_max_year INTEGER,
    timeline_length INTEGER,
    normalized_start REAL,
    normalized_end REAL,
    normalized_center REAL,
    years_with_evidence_count INTEGER,
    year_continuity_score REAL,
    gap_count INTEGER,
    max_gap_length INTEGER,
    prefix_support_score REAL,
    suffix_support_score REAL,
    coverage_density_score REAL,
    density_balance_score REAL,
    avg_chunk_count_per_year REAL,
    max_chunk_count_in_year INTEGER,
    min_chunk_count_in_year INTEGER,
    graph_feature_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (candidate_window_id, graph_feature_version)
);

CREATE INDEX IF NOT EXISTS idx_m0_graph_ticker_year ON m0_graph.ticker_year_evidence(ticker, year);
CREATE INDEX IF NOT EXISTS idx_m0_graph_candidate_cov_input ON m0_graph.candidate_year_coverage(m0_input_id);
CREATE INDEX IF NOT EXISTS idx_m0_graph_candidate_features_input ON m0_graph.candidate_graph_features(m0_input_id);
CREATE INDEX IF NOT EXISTS idx_m0_graph_candidate_features_type ON m0_graph.candidate_graph_features(window_type);
