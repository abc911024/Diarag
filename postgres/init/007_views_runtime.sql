CREATE OR REPLACE VIEW runtime.v_qa_content_coverage_summary AS
SELECT
    rewrite_type,
    m0_input_setting,
    coverage_status,
    COUNT(*) AS count,
    AVG(coverage_ratio) AS avg_coverage_ratio,
    AVG(matched_chunk_count) AS avg_matched_chunk_count
FROM runtime.qa_content_coverage
GROUP BY rewrite_type, m0_input_setting, coverage_status;

CREATE OR REPLACE VIEW runtime.v_m0_agent_results AS
SELECT
    ar.agent_run_id,
    ar.method,
    ar.m0_input_id,
    ar.bridge_id,
    br.rewritten_query,
    ar.ticker,
    ar.rewrite_type,
    ar.m0_input_setting,
    ar.predicted_range,
    am.gold_range,
    am.acceptable_ranges,
    am.overlap_score,
    am.mean_boundary_error,
    am.acceptable_accuracy,
    am.exact_match_accuracy,
    am.coverage_status,
    am.can_run_m1_retrieval
FROM runtime.m0_agent_runs ar
LEFT JOIN runtime.m0_agent_metrics am ON am.agent_run_id = ar.agent_run_id
JOIN qa.bridge_records br ON br.bridge_id = ar.bridge_id
JOIN qa.temporal_annotations ta ON ta.bridge_id = ar.bridge_id;

CREATE OR REPLACE VIEW runtime.v_m0_agent_summary AS
SELECT
    ar.method,
    ar.rewrite_type,
    ar.m0_input_setting,
    COUNT(*) AS count,
    AVG(am.overlap_score) AS avg_overlap_score,
    AVG(am.mean_boundary_error) AS avg_mean_boundary_error,
    AVG(CASE WHEN am.acceptable_accuracy THEN 1.0 ELSE 0.0 END) AS acceptable_accuracy_rate,
    AVG(CASE WHEN am.exact_match_accuracy THEN 1.0 ELSE 0.0 END) AS exact_match_rate,
    AVG(CASE WHEN am.can_run_m1_retrieval THEN 1.0 ELSE 0.0 END) AS m1_retrievable_rate
FROM runtime.m0_agent_runs ar
LEFT JOIN runtime.m0_agent_metrics am ON am.agent_run_id = ar.agent_run_id
GROUP BY ar.method, ar.rewrite_type, ar.m0_input_setting;

CREATE OR REPLACE VIEW runtime.v_probe_summary AS
SELECT
    probe_method,
    coverage_status,
    recommended_action,
    COUNT(*) AS count,
    AVG(matched_chunk_count) AS avg_matched_chunk_count
FROM runtime.m0_evidence_probes
GROUP BY probe_method, coverage_status, recommended_action;
