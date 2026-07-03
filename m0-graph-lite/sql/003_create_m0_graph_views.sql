CREATE OR REPLACE VIEW m0_graph.v_ticker_year_timeline_summary AS
SELECT
    ticker,
    build_version,
    MIN(year) AS min_year,
    MAX(year) AS max_year,
    COUNT(*) AS timeline_years,
    SUM(CASE WHEN has_evidence THEN 1 ELSE 0 END) AS years_with_evidence,
    SUM(chunk_count) AS total_chunks,
    AVG(chunk_count) AS avg_chunk_count,
    AVG(CASE WHEN has_evidence THEN 1.0 ELSE 0.0 END) AS evidence_coverage_ratio
FROM m0_graph.ticker_year_evidence
GROUP BY ticker, build_version;

CREATE OR REPLACE VIEW m0_graph.v_candidate_graph_feature_summary AS
SELECT
    window_type,
    graph_feature_version,
    COUNT(*) AS count,
    AVG(window_length) AS avg_window_length,
    AVG(year_continuity_score) AS avg_year_continuity_score,
    AVG(gap_count) AS avg_gap_count,
    AVG(max_gap_length) AS avg_max_gap_length,
    AVG(prefix_support_score) AS avg_prefix_support_score,
    AVG(suffix_support_score) AS avg_suffix_support_score,
    AVG(density_balance_score) AS avg_density_balance_score,
    AVG(coverage_density_score) AS avg_coverage_density_score
FROM m0_graph.candidate_graph_features
GROUP BY window_type, graph_feature_version;

CREATE OR REPLACE VIEW m0_graph.v_candidate_features_by_type AS
SELECT
    br.rewrite_type,
    gf.window_type,
    gf.graph_feature_version,
    COUNT(*) AS count,
    AVG(gf.year_continuity_score) AS avg_year_continuity_score,
    AVG(gf.gap_count) AS avg_gap_count,
    AVG(gf.prefix_support_score) AS avg_prefix_support_score,
    AVG(gf.suffix_support_score) AS avg_suffix_support_score,
    AVG(gf.normalized_center) AS avg_normalized_center,
    AVG(gf.density_balance_score) AS avg_density_balance_score
FROM m0_graph.candidate_graph_features gf
JOIN qa.bridge_records br ON br.bridge_id = gf.bridge_id
GROUP BY br.rewrite_type, gf.window_type, gf.graph_feature_version;
