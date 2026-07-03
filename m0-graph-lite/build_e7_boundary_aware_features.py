#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

from boundary_signal_utils import (
    compute_boundary_contrast,
    compute_inside_coherence,
    compute_intent_fit,
    compute_length_fit,
    compute_trend_coherence,
    compute_trend_signals,
    detect_relative_intent,
    safe_mean,
    safe_std,
)
from db_utils import add_db_args, connect_db, ensure_output_dirs, execute_sql_file, fetch_dataframe, write_dataframe_csv


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def apply_schema(conn) -> None:
    base = Path(__file__).resolve().parent / "sql"
    for name in ["006_create_boundary_aware_tables.sql", "007_create_e7_qualification_tables.sql"]:
        execute_sql_file(conn, base / name)
        print(f"applied {base / name}")


def clear_existing(conn, boundary_build_version: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM m0_graph.candidate_boundary_features WHERE build_version = %s", (boundary_build_version,))
        cur.execute("DELETE FROM m0_graph.year_trend_signals WHERE build_version = %s", (boundary_build_version,))
        cur.execute("DELETE FROM m0_graph.e7_candidate_boundary_diagnostics WHERE boundary_build_version = %s", (boundary_build_version,))
    conn.commit()


def selected_inputs(conn, method: str, setting: str, limit: int) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT ar.agent_run_id, ar.m0_input_id, ar.bridge_id, br.ticker,
                   br.rewritten_query AS query_text, br.rewrite_type
            FROM runtime.m0_agent_runs ar
            JOIN qa.bridge_records br ON br.bridge_id = ar.bridge_id
            WHERE ar.method = %s AND ar.m0_input_setting = %s
            ORDER BY ar.created_at DESC
            LIMIT %s
            """,
            (method, setting, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_candidates(conn, agent_run_id: str) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM runtime.m0_candidate_windows WHERE candidate_window_id LIKE %s ORDER BY candidate_window_id",
            (f"{agent_run_id}__cand_%",),
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_raw_chunks(conn, m0_input_id: str, year: int, source_build_version: str) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT chunk_id, chunk_text, hybrid_relevance_score
            FROM m0_graph.hybrid_query_relevant_chunks
            WHERE m0_input_id = %s AND year = %s AND build_version = %s
            ORDER BY rank_in_year
            """,
            (m0_input_id, year, source_build_version),
        )
        return [dict(r) for r in cur.fetchall()]


def aggregate_relevance(raw_chunks: list[dict[str, Any]], routed_chunks: list[dict[str, Any]]) -> float:
    scores = [float(c.get("hybrid_relevance_score") or 0.0) for c in routed_chunks]
    if not scores:
        return 0.0
    return 0.50 * max(scores) + 0.30 * safe_mean(scores) + 0.20 * min(1.0, len(scores) / max(1, len(raw_chunks)))


def fetch_routed_chunks(conn, m0_input_id: str, year: int, source_build_version: str, qualification_build_version: str, usage_column: str) -> list[dict[str, Any]]:
    allowed = {
        "use_for_inside_relevance",
        "use_for_evidence_coverage",
        "use_for_boundary_contrast",
        "use_for_trend_coherence",
    }
    if usage_column not in allowed:
        raise ValueError(f"Unsupported E7b usage column: {usage_column}")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT h.chunk_id, h.chunk_text, h.hybrid_relevance_score
            FROM m0_graph.hybrid_query_relevant_chunks h
            JOIN m0_graph.chunk_temporal_qualifications q
              ON q.m0_input_id = h.m0_input_id
             AND q.year = h.year
             AND q.chunk_id = h.chunk_id
             AND q.source_build_version = h.build_version
            WHERE h.m0_input_id = %s
              AND h.year = %s
              AND h.build_version = %s
              AND q.qualification_build_version = %s
              AND q.{usage_column} = TRUE
            ORDER BY h.rank_in_year
            """,
            (m0_input_id, year, source_build_version, qualification_build_version),
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_label_counts(conn, m0_input_id: str, years: list[int], source_build_version: str, qualification_build_version: str) -> dict[str, int]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) AS raw,
                COUNT(*) FILTER (WHERE use_for_inside_relevance) AS inside,
                COUNT(*) FILTER (WHERE use_for_evidence_coverage) AS coverage,
                COUNT(*) FILTER (WHERE use_for_boundary_contrast) AS boundary,
                COUNT(*) FILTER (WHERE use_for_trend_coherence) AS trend,
                COUNT(*) FILTER (WHERE qualification_label IN ('boundary_temporal_evidence', 'general_temporal_evidence', 'qualified_temporal_evidence')) AS qualified,
                COUNT(*) FILTER (WHERE evidence_label = 'boundary_temporal_evidence') AS boundary_label,
                COUNT(*) FILTER (WHERE evidence_label = 'general_temporal_evidence') AS general_label,
                COUNT(*) FILTER (WHERE qualification_label = 'supporting_context') AS supporting,
                COUNT(*) FILTER (WHERE qualification_label = 'low_value_noise') AS noise,
                COUNT(DISTINCT year) FILTER (WHERE use_for_evidence_coverage) AS qualified_years,
                COUNT(DISTINCT year) FILTER (WHERE use_for_boundary_contrast) AS boundary_years
            FROM m0_graph.chunk_temporal_qualifications
            WHERE m0_input_id = %s
              AND year = ANY(%s)
              AND source_build_version = %s
              AND qualification_build_version = %s
            """,
            (m0_input_id, years, source_build_version, qualification_build_version),
        )
        row = dict(cur.fetchone())
        return {k: int(row.get(k) or 0) for k in (
            "raw", "inside", "coverage", "boundary", "trend", "qualified",
            "boundary_label", "general_label", "supporting", "noise",
            "qualified_years", "boundary_years",
        )}


def insert_year_trend_signal(
    conn,
    item: dict[str, Any],
    year: int,
    raw_chunks: list[dict[str, Any]],
    inside_chunks: list[dict[str, Any]],
    trend_chunks: list[dict[str, Any]],
    build_version: str,
) -> None:
    signals = compute_trend_signals([c["chunk_text"] for c in trend_chunks])
    relevance = aggregate_relevance(raw_chunks, inside_chunks)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO m0_graph.year_trend_signals (
                m0_input_id, bridge_id, ticker, query_text, rewrite_type, year,
                hybrid_year_relevance_score, year_has_hybrid_relevant_evidence,
                trend_signal_score, up_signal_score, down_signal_score, stable_signal_score,
                volatile_signal_score, dominant_trend_direction, top_trend_terms, top_chunk_ids,
                build_version
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
            ON CONFLICT (m0_input_id, year, build_version) DO UPDATE SET
                hybrid_year_relevance_score = EXCLUDED.hybrid_year_relevance_score,
                year_has_hybrid_relevant_evidence = EXCLUDED.year_has_hybrid_relevant_evidence,
                trend_signal_score = EXCLUDED.trend_signal_score,
                dominant_trend_direction = EXCLUDED.dominant_trend_direction,
                top_trend_terms = EXCLUDED.top_trend_terms,
                top_chunk_ids = EXCLUDED.top_chunk_ids
            """,
            (
                item["m0_input_id"], item["bridge_id"], item["ticker"], item["query_text"], item["rewrite_type"],
                year, relevance, bool(inside_chunks), signals["trend_signal_score"],
                signals["up_signal_score"], signals["down_signal_score"], signals["stable_signal_score"],
                signals["volatile_signal_score"], signals["dominant_trend_direction"],
                dump(signals["top_trend_terms"]), dump([c["chunk_id"] for c in trend_chunks]), build_version,
            ),
        )


def fetch_year_trends(conn, m0_input_id: str, build_version: str) -> dict[int, dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM m0_graph.year_trend_signals WHERE m0_input_id = %s AND build_version = %s", (m0_input_id, build_version))
        return {int(r["year"]): dict(r) for r in cur.fetchall()}


def insert_candidate_boundary_feature(
    conn,
    item: dict[str, Any],
    candidate: dict[str, Any],
    all_years: list[int],
    trends: dict[int, dict[str, Any]],
    boundary_relevance_by_year: dict[int, float],
    counts: dict[str, int],
    context_size: int,
    source_build_version: str,
    qualification_build_version: str,
    boundary_build_version: str,
) -> None:
    start = int(candidate["start_year"])
    end = int(candidate["end_year"])
    inside_years = list(range(start, end + 1))
    timeline_min = min(all_years)
    timeline_max = max(all_years)
    timeline_len = timeline_max - timeline_min + 1
    relevance_by_year = {year: float(trends.get(year, {}).get("hybrid_year_relevance_score") or 0.0) for year in all_years}
    trend_score_by_year = {year: float(trends.get(year, {}).get("trend_signal_score") or 0.0) for year in all_years}
    direction_by_year = {year: str(trends.get(year, {}).get("dominant_trend_direction") or "unknown") for year in all_years}
    inside_values = [relevance_by_year.get(year, 0.0) for year in inside_years]
    inside_mean = safe_mean(inside_values)
    inside_std = safe_std(inside_values)
    inside_coherence = compute_inside_coherence(inside_values)
    contrast = compute_boundary_contrast(all_years, boundary_relevance_by_year, start, end, context_size)
    trend_values = [trend_score_by_year.get(year, 0.0) for year in inside_years]
    trend = compute_trend_coherence([direction_by_year.get(year, "unknown") for year in inside_years])
    length = end - start + 1
    center = (start + end) / 2
    normalized_center = (center - timeline_min) / max(1, timeline_len - 1)
    relative_intent = detect_relative_intent(item["query_text"], item["rewrite_type"])
    intent_fit = compute_intent_fit(relative_intent, candidate.get("window_type") or "", normalized_center, length, timeline_len)
    length_fit = compute_length_fit(length)
    boundary_score_raw = (
        0.15 * intent_fit
        + 0.20 * inside_mean
        + 0.15 * inside_coherence
        + 0.20 * contrast["boundary_contrast"]
        + 0.20 * trend["trend_coherence"]
        + 0.10 * length_fit
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO m0_graph.candidate_boundary_features (
                candidate_window_id, m0_input_id, bridge_id, ticker, query_text, rewrite_type,
                start_year, end_year, window_type, window_length, timeline_min_year,
                timeline_max_year, timeline_length, inside_relevance_mean, inside_relevance_std,
                inside_coherence, left_context_relevance, right_context_relevance,
                left_boundary_contrast, right_boundary_contrast, boundary_contrast,
                trend_signal_mean, trend_coherence, dominant_window_trend_direction,
                intent_fit, length_fit, boundary_score_raw, context_size, build_version
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (candidate_window_id, build_version) DO UPDATE SET
                inside_relevance_mean = EXCLUDED.inside_relevance_mean,
                inside_coherence = EXCLUDED.inside_coherence,
                boundary_contrast = EXCLUDED.boundary_contrast,
                trend_coherence = EXCLUDED.trend_coherence,
                boundary_score_raw = EXCLUDED.boundary_score_raw
            """,
            (
                candidate["candidate_window_id"], item["m0_input_id"], item["bridge_id"], item["ticker"],
                item["query_text"], item["rewrite_type"], start, end, candidate.get("window_type"), length,
                timeline_min, timeline_max, timeline_len, inside_mean, inside_std, inside_coherence,
                contrast["left_context_relevance"], contrast["right_context_relevance"],
                contrast["left_boundary_contrast"], contrast["right_boundary_contrast"], contrast["boundary_contrast"],
                safe_mean(trend_values), trend["trend_coherence"], trend["dominant_window_trend_direction"],
                intent_fit, length_fit, boundary_score_raw, context_size, boundary_build_version,
            ),
        )
        cur.execute(
            """
            INSERT INTO m0_graph.e7_candidate_boundary_diagnostics (
                candidate_window_id, m0_input_id, bridge_id, ticker, query_text, rewrite_type,
                start_year, end_year, window_type, window_length, raw_chunk_count,
                inside_relevance_chunk_count, evidence_coverage_chunk_count,
                boundary_contrast_chunk_count, trend_coherence_chunk_count,
                qualified_chunk_count, boundary_temporal_evidence_count,
                general_temporal_evidence_count, supporting_context_count,
                low_value_noise_count, qualified_evidence_year_count,
                boundary_evidence_year_count, qualified_evidence_year_coverage,
                boundary_evidence_year_coverage, background_noise_ratio,
                source_build_version, qualification_build_version,
                boundary_build_version
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (candidate_window_id, boundary_build_version) DO UPDATE SET
                raw_chunk_count = EXCLUDED.raw_chunk_count,
                inside_relevance_chunk_count = EXCLUDED.inside_relevance_chunk_count,
                evidence_coverage_chunk_count = EXCLUDED.evidence_coverage_chunk_count,
                boundary_contrast_chunk_count = EXCLUDED.boundary_contrast_chunk_count,
                trend_coherence_chunk_count = EXCLUDED.trend_coherence_chunk_count,
                qualified_chunk_count = EXCLUDED.qualified_chunk_count,
                boundary_temporal_evidence_count = EXCLUDED.boundary_temporal_evidence_count,
                general_temporal_evidence_count = EXCLUDED.general_temporal_evidence_count,
                supporting_context_count = EXCLUDED.supporting_context_count,
                low_value_noise_count = EXCLUDED.low_value_noise_count,
                qualified_evidence_year_count = EXCLUDED.qualified_evidence_year_count,
                boundary_evidence_year_count = EXCLUDED.boundary_evidence_year_count,
                qualified_evidence_year_coverage = EXCLUDED.qualified_evidence_year_coverage,
                boundary_evidence_year_coverage = EXCLUDED.boundary_evidence_year_coverage,
                background_noise_ratio = EXCLUDED.background_noise_ratio
            """,
            (
                candidate["candidate_window_id"], item["m0_input_id"], item["bridge_id"], item["ticker"],
                item["query_text"], item["rewrite_type"], start, end, candidate.get("window_type"), length,
                counts["raw"], counts["inside"], counts["coverage"], counts["boundary"], counts["trend"],
                counts["qualified"], counts["boundary_label"], counts["general_label"],
                counts["supporting"], counts["noise"], counts["qualified_years"], counts["boundary_years"],
                counts["qualified_years"] / max(1, length), counts["boundary_years"] / max(1, length),
                (counts["supporting"] + counts["noise"]) / max(1, counts["raw"]),
                source_build_version, qualification_build_version, boundary_build_version,
            ),
        )


def build_features(conn, inputs: list[dict[str, Any]], source_build_version: str, qualification_build_version: str, boundary_build_version: str, context_size: int) -> None:
    for item in inputs:
        candidates = fetch_candidates(conn, item["agent_run_id"])
        if not candidates:
            continue
        timeline_min = min(int(c["start_year"]) for c in candidates)
        timeline_max = max(int(c["end_year"]) for c in candidates)
        all_years = list(range(timeline_min, timeline_max + 1))
        boundary_relevance_by_year: dict[int, float] = {}
        for year in all_years:
            raw_chunks = fetch_raw_chunks(conn, item["m0_input_id"], year, source_build_version)
            inside_chunks = fetch_routed_chunks(conn, item["m0_input_id"], year, source_build_version, qualification_build_version, "use_for_inside_relevance")
            boundary_chunks = fetch_routed_chunks(conn, item["m0_input_id"], year, source_build_version, qualification_build_version, "use_for_boundary_contrast")
            trend_chunks = fetch_routed_chunks(conn, item["m0_input_id"], year, source_build_version, qualification_build_version, "use_for_trend_coherence")
            boundary_relevance_by_year[year] = aggregate_relevance(raw_chunks, boundary_chunks)
            insert_year_trend_signal(conn, item, year, raw_chunks, inside_chunks, trend_chunks, boundary_build_version)
        trends = fetch_year_trends(conn, item["m0_input_id"], boundary_build_version)
        for candidate in candidates:
            inside_years = list(range(int(candidate["start_year"]), int(candidate["end_year"]) + 1))
            counts = fetch_label_counts(conn, item["m0_input_id"], inside_years, source_build_version, qualification_build_version)
            insert_candidate_boundary_feature(
                conn, item, candidate, all_years, trends, boundary_relevance_by_year, counts, context_size,
                source_build_version, qualification_build_version, boundary_build_version,
            )
        conn.commit()


def export_outputs(conn) -> None:
    exports = {
        "e7_candidate_boundary_diagnostics.csv": "SELECT * FROM m0_graph.e7_candidate_boundary_diagnostics ORDER BY m0_input_id, candidate_window_id",
        "e7_candidate_boundary_diagnostic_summary.csv": "SELECT * FROM m0_graph.v_e7_candidate_boundary_diagnostic_summary ORDER BY rewrite_type, window_type",
        "e7_candidate_boundary_features.csv": "SELECT * FROM m0_graph.candidate_boundary_features WHERE build_version LIKE 'e7_%' ORDER BY m0_input_id, candidate_window_id",
    }
    for name, sql in exports.items():
        write_dataframe_csv(fetch_dataframe(conn, sql), Path("m0-graph-lite/outputs/csv") / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_db_args(parser)
    parser.add_argument("--method", default="vector_probe_agent_v1")
    parser.add_argument("--m0-input-setting", default="query_with_corpus_metadata")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--context-size", type=int, default=2)
    parser.add_argument("--source-build-version", default="graph_lite_v2_hybrid")
    parser.add_argument("--qualification-build-version", default="e7b_rule_v1")
    parser.add_argument("--boundary-build-version", default="e7b_boundary_aware_qualification_v1")
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    with connect_db(args) as conn:
        apply_schema(conn)
        if args.clear_existing:
            clear_existing(conn, args.boundary_build_version)
        inputs = selected_inputs(conn, args.method, args.m0_input_setting, args.limit)
        build_features(conn, inputs, args.source_build_version, args.qualification_build_version, args.boundary_build_version, args.context_size)
        export_outputs(conn)


if __name__ == "__main__":
    main()
