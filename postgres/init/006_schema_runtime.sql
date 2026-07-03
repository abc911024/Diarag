CREATE SCHEMA IF NOT EXISTS runtime;

CREATE TABLE IF NOT EXISTS runtime.qa_content_coverage (
    coverage_id TEXT PRIMARY KEY,
    bridge_id TEXT REFERENCES qa.bridge_records(bridge_id),
    m0_input_id TEXT REFERENCES qa.m0_inputs(m0_input_id),
    ticker TEXT,
    rewritten_query TEXT,
    rewrite_type TEXT,
    m0_input_setting TEXT,
    gold_start_year INTEGER,
    gold_end_year INTEGER,
    gold_years JSONB,
    content_available_years JSONB,
    covered_gold_years JSONB,
    missing_gold_years JSONB,
    matched_doc_count INTEGER,
    matched_chunk_count INTEGER,
    coverage_ratio REAL,
    coverage_status TEXT CHECK (coverage_status IN (
        'full_gold_coverage',
        'partial_gold_coverage',
        'no_gold_coverage',
        'no_content_for_ticker',
        'missing_gold_target',
        'unknown'
    )),
    can_run_m1_retrieval BOOLEAN,
    can_run_temporal_retrieval BOOLEAN,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runtime.m0_candidate_windows (
    candidate_window_id TEXT PRIMARY KEY,
    m0_input_id TEXT REFERENCES qa.m0_inputs(m0_input_id),
    bridge_id TEXT REFERENCES qa.bridge_records(bridge_id),
    ticker TEXT,
    rewrite_type TEXT,
    m0_input_setting TEXT,
    window_type TEXT,
    start_year INTEGER,
    end_year INTEGER,
    candidate_range JSONB,
    generation_method TEXT,
    is_visible_to_model BOOLEAN DEFAULT TRUE,
    uses_gold_label BOOLEAN DEFAULT FALSE,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE runtime.m0_candidate_windows ADD COLUMN IF NOT EXISTS priority_hint TEXT;
ALTER TABLE runtime.m0_candidate_windows ADD COLUMN IF NOT EXISTS relative_intent TEXT;
ALTER TABLE runtime.m0_candidate_windows ADD COLUMN IF NOT EXISTS candidate_version TEXT;
ALTER TABLE runtime.m0_candidate_windows ADD COLUMN IF NOT EXISTS use_as_final_candidate BOOLEAN DEFAULT TRUE;
ALTER TABLE runtime.m0_candidate_windows ADD COLUMN IF NOT EXISTS score REAL;
ALTER TABLE runtime.m0_candidate_windows ADD COLUMN IF NOT EXISTS score_components JSONB;

CREATE TABLE IF NOT EXISTS runtime.m0_candidate_oracle_analysis (
    analysis_id TEXT PRIMARY KEY,
    m0_input_id TEXT,
    bridge_id TEXT,
    rewrite_type TEXT,
    m0_input_setting TEXT,
    total_candidates INTEGER,
    has_exact_candidate BOOLEAN,
    has_acceptable_candidate BOOLEAN,
    best_candidate_window_id TEXT,
    best_candidate_range JSONB,
    best_candidate_overlap REAL,
    best_candidate_mean_boundary_error REAL,
    best_candidate_acceptable_accuracy BOOLEAN,
    selected_candidate_window_id TEXT NULL,
    selected_candidate_range JSONB NULL,
    selected_overlap REAL NULL,
    selected_acceptable_accuracy BOOLEAN NULL,
    candidate_generation_version TEXT,
    scoring_version TEXT,
    method TEXT,
    diagnosis TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runtime.m0_evidence_probes (
    probe_id TEXT PRIMARY KEY,
    candidate_window_id TEXT REFERENCES runtime.m0_candidate_windows(candidate_window_id),
    m0_input_id TEXT REFERENCES qa.m0_inputs(m0_input_id),
    query_text TEXT,
    ticker TEXT,
    start_year INTEGER,
    end_year INTEGER,
    top_k INTEGER,
    retrieved_chunk_ids JSONB,
    retrieved_doc_ids JSONB,
    retrieved_years JSONB,
    years_with_evidence JSONB,
    missing_years JSONB,
    matched_chunk_count INTEGER,
    matched_doc_count INTEGER,
    evidence_density JSONB,
    coverage_status TEXT,
    recommended_action TEXT,
    probe_method TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runtime.m0_agent_runs (
    agent_run_id TEXT PRIMARY KEY,
    m0_input_id TEXT REFERENCES qa.m0_inputs(m0_input_id),
    bridge_id TEXT REFERENCES qa.bridge_records(bridge_id),
    method TEXT,
    query_text TEXT,
    ticker TEXT,
    rewrite_type TEXT,
    m0_input_setting TEXT,
    predicted_start_year INTEGER,
    predicted_end_year INTEGER,
    predicted_range JSONB,
    selected_candidate_window_id TEXT NULL,
    confidence REAL,
    decision_status TEXT,
    rationale TEXT,
    tool_trace JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runtime.m0_agent_metrics (
    agent_run_id TEXT PRIMARY KEY REFERENCES runtime.m0_agent_runs(agent_run_id),
    gold_range JSONB,
    acceptable_ranges JSONB,
    overlap_score REAL,
    start_boundary_error INTEGER,
    end_boundary_error INTEGER,
    mean_boundary_error REAL,
    acceptable_accuracy BOOLEAN,
    exact_match_accuracy BOOLEAN,
    prediction_length_error INTEGER,
    over_extension INTEGER,
    under_extension INTEGER,
    coverage_status TEXT,
    can_run_m1_retrieval BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runtime_coverage_bridge_id ON runtime.qa_content_coverage(bridge_id);
CREATE INDEX IF NOT EXISTS idx_runtime_coverage_m0_input_id ON runtime.qa_content_coverage(m0_input_id);
CREATE INDEX IF NOT EXISTS idx_runtime_coverage_ticker ON runtime.qa_content_coverage(ticker);
CREATE INDEX IF NOT EXISTS idx_runtime_coverage_status ON runtime.qa_content_coverage(coverage_status);
CREATE INDEX IF NOT EXISTS idx_runtime_candidate_windows_m0_input_id ON runtime.m0_candidate_windows(m0_input_id);
CREATE INDEX IF NOT EXISTS idx_runtime_candidate_windows_ticker ON runtime.m0_candidate_windows(ticker);
CREATE INDEX IF NOT EXISTS idx_runtime_candidate_windows_candidate_version ON runtime.m0_candidate_windows(candidate_version);
CREATE INDEX IF NOT EXISTS idx_runtime_candidate_oracle_m0_input_id ON runtime.m0_candidate_oracle_analysis(m0_input_id);
CREATE INDEX IF NOT EXISTS idx_runtime_candidate_oracle_diagnosis ON runtime.m0_candidate_oracle_analysis(diagnosis);
CREATE INDEX IF NOT EXISTS idx_runtime_evidence_probes_m0_input_id ON runtime.m0_evidence_probes(m0_input_id);
CREATE INDEX IF NOT EXISTS idx_runtime_evidence_probes_coverage_status ON runtime.m0_evidence_probes(coverage_status);
CREATE INDEX IF NOT EXISTS idx_runtime_agent_runs_m0_input_id ON runtime.m0_agent_runs(m0_input_id);
CREATE INDEX IF NOT EXISTS idx_runtime_agent_runs_method ON runtime.m0_agent_runs(method);
CREATE INDEX IF NOT EXISTS idx_runtime_agent_metrics_acceptable_accuracy ON runtime.m0_agent_metrics(acceptable_accuracy);
