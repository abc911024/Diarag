CREATE TABLE IF NOT EXISTS qa.raw_qa_items (
    source_id TEXT PRIMARY KEY,
    source_dataset TEXT DEFAULT 'DQABench',
    raw_json JSONB NOT NULL,
    ticker TEXT,
    stock_name TEXT,
    original_question TEXT,
    temporal_context_label TEXT,
    query_type TEXT,
    query_type_id TEXT,
    generation_score INTEGER,
    correct_answer_key TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS qa.bridge_candidates (
    source_id TEXT PRIMARY KEY REFERENCES qa.raw_qa_items(source_id),
    is_bridge_candidate BOOLEAN NOT NULL,
    candidate_reason TEXT,
    reject_reason TEXT,
    duration_days INTEGER,
    detected_diachronic_intent BOOLEAN,
    detected_factoid BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS qa.bridge_records (
    bridge_id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES qa.raw_qa_items(source_id),
    source_dataset TEXT DEFAULT 'DQABench',
    original_query TEXT,
    pseudo_query TEXT,
    rewritten_query TEXT,
    rewrite_type TEXT,
    rewrite_status TEXT,
    rewrite_reason TEXT,
    rewrite_quality_status TEXT,
    rewrite_quality_flags JSONB,
    ticker TEXT,
    stock_name TEXT,
    metadata_used_for_rewrite BOOLEAN DEFAULT FALSE,
    annotation_uses_metadata BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS qa.temporal_annotations (
    bridge_id TEXT PRIMARY KEY REFERENCES qa.bridge_records(bridge_id),
    gold_start_date DATE,
    gold_end_date DATE,
    gold_start_year INTEGER,
    gold_end_year INTEGER,
    gold_range JSONB,
    acceptable_ranges JSONB,
    temporal_scope_type TEXT,
    source_temporal_label TEXT,
    anchor_year INTEGER NULL,
    anchor_direction TEXT NULL,
    annotation_source TEXT DEFAULT 'plot_time_bounds'
);

CREATE TABLE IF NOT EXISTS qa.entity_coverage (
    entity_id TEXT PRIMARY KEY,
    ticker TEXT,
    stock_name TEXT,
    min_year INTEGER,
    max_year INTEGER,
    available_years JSONB,
    coverage_source TEXT DEFAULT 'derived_from_qa_gold_coverage_for_m0_experiment'
);

CREATE TABLE IF NOT EXISTS qa.m0_inputs (
    m0_input_id TEXT PRIMARY KEY,
    bridge_id TEXT REFERENCES qa.bridge_records(bridge_id),
    m0_input_setting TEXT CHECK (m0_input_setting IN ('query_only', 'query_with_corpus_metadata', 'query_with_verification')),
    m0_input_json JSONB NOT NULL,
    rewrite_type TEXT,
    temporal_need_type TEXT DEFAULT 'historical_trend',
    evidence_policy TEXT DEFAULT 'broad_historical_coverage',
    input_contains_gold_answer BOOLEAN DEFAULT FALSE,
    input_difficulty_note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS qa.m0_predictions (
    prediction_id TEXT PRIMARY KEY,
    m0_input_id TEXT REFERENCES qa.m0_inputs(m0_input_id),
    method TEXT,
    predicted_start_year INTEGER,
    predicted_end_year INTEGER,
    predicted_range JSONB,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS qa.m0_metrics (
    prediction_id TEXT PRIMARY KEY REFERENCES qa.m0_predictions(prediction_id),
    overlap_score REAL,
    start_boundary_error INTEGER,
    end_boundary_error INTEGER,
    mean_boundary_error REAL,
    acceptable_accuracy BOOLEAN,
    exact_match_accuracy BOOLEAN,
    prediction_length_error INTEGER,
    over_extension INTEGER,
    under_extension INTEGER
);

CREATE TABLE IF NOT EXISTS qa.qa_pipeline_runs (
    run_id TEXT PRIMARY KEY,
    input_path TEXT,
    output_dir TEXT,
    total_raw_items INTEGER,
    accepted_candidates INTEGER,
    rejected_candidates INTEGER,
    bridge_records INTEGER,
    m0_inputs INTEGER,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
