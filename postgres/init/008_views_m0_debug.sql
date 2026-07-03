CREATE OR REPLACE VIEW runtime.v_selected_window_type_summary AS
SELECT
    ar.method,
    ar.rewrite_type,
    ar.m0_input_setting,
    cw.window_type,
    COUNT(*) AS count,
    AVG(am.overlap_score) AS avg_overlap_score,
    AVG(am.mean_boundary_error) AS avg_boundary_error,
    AVG(CASE WHEN am.acceptable_accuracy THEN 1.0 ELSE 0.0 END) AS acceptable_accuracy_rate,
    AVG(ar.predicted_end_year - ar.predicted_start_year + 1) AS avg_predicted_length
FROM runtime.m0_agent_runs ar
LEFT JOIN runtime.m0_agent_metrics am ON am.agent_run_id = ar.agent_run_id
LEFT JOIN runtime.m0_candidate_windows cw ON cw.candidate_window_id = ar.selected_candidate_window_id
GROUP BY ar.method, ar.rewrite_type, ar.m0_input_setting, cw.window_type;

CREATE OR REPLACE VIEW runtime.v_candidate_oracle_summary AS
SELECT
    candidate_generation_version,
    scoring_version,
    rewrite_type,
    m0_input_setting,
    COUNT(*) AS count,
    AVG(CASE WHEN has_exact_candidate THEN 1.0 ELSE 0.0 END) AS candidate_exact_rate,
    AVG(CASE WHEN has_acceptable_candidate THEN 1.0 ELSE 0.0 END) AS candidate_acceptable_rate,
    AVG(best_candidate_overlap) AS avg_best_candidate_overlap,
    AVG(selected_overlap) AS avg_selected_overlap,
    AVG(CASE WHEN selected_acceptable_accuracy THEN 1.0 ELSE 0.0 END) AS avg_selected_acceptable_accuracy
FROM runtime.m0_candidate_oracle_analysis
GROUP BY candidate_generation_version, scoring_version, rewrite_type, m0_input_setting;

CREATE OR REPLACE VIEW runtime.v_m0_length_bias AS
SELECT
    ar.method,
    ar.rewrite_type,
    ar.m0_input_setting,
    AVG(ar.predicted_end_year - ar.predicted_start_year + 1) AS avg_pred_len,
    AVG(((am.gold_range ->> 1)::INTEGER - (am.gold_range ->> 0)::INTEGER + 1)) AS avg_gold_len,
    AVG((ar.predicted_end_year - ar.predicted_start_year + 1) - ((am.gold_range ->> 1)::INTEGER - (am.gold_range ->> 0)::INTEGER + 1)) AS avg_length_diff
FROM runtime.m0_agent_runs ar
JOIN runtime.m0_agent_metrics am ON am.agent_run_id = ar.agent_run_id
WHERE ar.predicted_start_year IS NOT NULL
  AND ar.predicted_end_year IS NOT NULL
  AND jsonb_typeof(am.gold_range) = 'array'
GROUP BY ar.method, ar.rewrite_type, ar.m0_input_setting;

CREATE OR REPLACE VIEW runtime.v_type_b_relative_error_summary AS
SELECT
    cw.relative_intent,
    cw.window_type AS selected_window_type,
    COUNT(*) AS count,
    AVG(am.overlap_score) AS avg_overlap,
    AVG(CASE WHEN am.acceptable_accuracy THEN 1.0 ELSE 0.0 END) AS acceptable_accuracy_rate
FROM runtime.m0_agent_runs ar
JOIN runtime.m0_agent_metrics am ON am.agent_run_id = ar.agent_run_id
LEFT JOIN runtime.m0_candidate_windows cw ON cw.candidate_window_id = ar.selected_candidate_window_id
WHERE ar.rewrite_type = 'B'
GROUP BY cw.relative_intent, cw.window_type;
