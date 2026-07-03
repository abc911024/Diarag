#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime

from db_config import add_connection_args, get_connection
from m0_agent_tools import (
    choose_best_window,
    choose_best_window_v2,
    compute_m0_metrics,
    diagnose_query,
    generate_candidate_windows,
    generate_candidate_windows_v2,
    get_content_available_years,
    get_entity_coverage,
    get_gold_target,
    get_m0_input,
    probe_content_metadata,
    score_candidate_window,
    score_candidate_window_v2,
    vector_probe_content,
)


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _range_equal(a, b) -> bool:
    return isinstance(a, list) and isinstance(b, list) and len(a) == 2 and len(b) == 2 and [int(a[0]), int(a[1])] == [int(b[0]), int(b[1])]


def store_oracle_analysis(conn, agent_run_id: str, m0_input: dict, scored: list[dict], selected: dict, metrics: dict, gold: dict, method: str, candidate_version: str, scoring_version: str) -> None:
    gold_range = gold.get("gold_range")
    acceptable_ranges = gold.get("acceptable_ranges") or []
    best_item = None
    has_exact = False
    has_acceptable = False
    for item in scored:
        cand = item["candidate"]
        cand_range = cand.get("candidate_range")
        cand_metrics = compute_m0_metrics(cand_range, gold_range, acceptable_ranges)
        item["oracle_metrics"] = cand_metrics
        has_exact = has_exact or bool(cand_metrics["exact_match_accuracy"])
        has_acceptable = has_acceptable or bool(cand_metrics["acceptable_accuracy"])
        if best_item is None:
            best_item = item
        else:
            prev = best_item["oracle_metrics"]
            if (
                cand_metrics["overlap_score"],
                -(cand_metrics["mean_boundary_error"] or 9999),
                cand_metrics["acceptable_accuracy"],
            ) > (
                prev["overlap_score"],
                -(prev["mean_boundary_error"] or 9999),
                prev["acceptable_accuracy"],
            ):
                best_item = item
    selected_range = selected.get("candidate_range")
    selected_acceptable = bool(metrics.get("acceptable_accuracy"))
    if not has_acceptable:
        diagnosis = "candidate_generation_bottleneck"
    elif not selected_acceptable:
        diagnosis = "scoring_bottleneck"
    elif selected_acceptable:
        diagnosis = "selected_good_candidate"
    else:
        diagnosis = "unknown"
    best_candidate = best_item["candidate"] if best_item else {}
    best_metrics = best_item.get("oracle_metrics", {}) if best_item else {}
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO runtime.m0_candidate_oracle_analysis (
                analysis_id, m0_input_id, bridge_id, rewrite_type, m0_input_setting,
                total_candidates, has_exact_candidate, has_acceptable_candidate,
                best_candidate_window_id, best_candidate_range, best_candidate_overlap,
                best_candidate_mean_boundary_error, best_candidate_acceptable_accuracy,
                selected_candidate_window_id, selected_candidate_range, selected_overlap,
                selected_acceptable_accuracy, candidate_generation_version, scoring_version,
                method, diagnosis
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (analysis_id) DO UPDATE SET
                total_candidates = EXCLUDED.total_candidates,
                has_exact_candidate = EXCLUDED.has_exact_candidate,
                has_acceptable_candidate = EXCLUDED.has_acceptable_candidate,
                best_candidate_window_id = EXCLUDED.best_candidate_window_id,
                best_candidate_range = EXCLUDED.best_candidate_range,
                best_candidate_overlap = EXCLUDED.best_candidate_overlap,
                best_candidate_mean_boundary_error = EXCLUDED.best_candidate_mean_boundary_error,
                best_candidate_acceptable_accuracy = EXCLUDED.best_candidate_acceptable_accuracy,
                selected_candidate_window_id = EXCLUDED.selected_candidate_window_id,
                selected_candidate_range = EXCLUDED.selected_candidate_range,
                selected_overlap = EXCLUDED.selected_overlap,
                selected_acceptable_accuracy = EXCLUDED.selected_acceptable_accuracy,
                diagnosis = EXCLUDED.diagnosis
            """,
            (
                f"{agent_run_id}__oracle", m0_input["m0_input_id"], m0_input["bridge_id"],
                m0_input["rewrite_type"], m0_input["m0_input_setting"], len(scored), has_exact,
                has_acceptable, best_candidate.get("candidate_window_id"), _dump(best_candidate.get("candidate_range")),
                best_metrics.get("overlap_score"), best_metrics.get("mean_boundary_error"),
                best_metrics.get("acceptable_accuracy"), selected.get("candidate_window_id"),
                _dump(selected_range), metrics.get("overlap_score"), selected_acceptable,
                candidate_version, scoring_version, method, diagnosis,
            ),
        )


def run_agent(
    conn,
    m0_input_id: str,
    method: str,
    top_k: int,
    allow_fallback: bool,
    agent_version: str = "v2",
    candidate_version: str = "v2",
    scoring_version: str = "v2",
    store_oracle: bool = True,
) -> dict:
    m0_input = get_m0_input(conn, m0_input_id)
    diagnosis = diagnose_query(m0_input)
    ticker = m0_input["ticker"]
    entity_coverage = get_entity_coverage(conn, ticker)
    content_years = get_content_available_years(conn, ticker)
    tool_trace = {
        "diagnosis": diagnosis,
        "entity_coverage": entity_coverage,
        "content_years": content_years,
        "candidates": [],
    }
    run_suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    agent_run_id = f"{m0_input_id}__{method}__{run_suffix}"
    query_text = m0_input.get("rewritten_query") or (m0_input.get("m0_input_json") or {}).get("rewritten_query") or ""
    if candidate_version == "v2":
        candidates = generate_candidate_windows_v2(m0_input, entity_coverage, content_years)
    else:
        candidates = generate_candidate_windows(m0_input, entity_coverage, content_years)
    scored = []
    with conn.cursor() as cur:
        if not entity_coverage or not candidates:
            cur.execute(
                """
                INSERT INTO runtime.m0_agent_runs (
                    agent_run_id, m0_input_id, bridge_id, method, query_text, ticker, rewrite_type,
                    m0_input_setting, decision_status, rationale, tool_trace
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (agent_run_id) DO NOTHING
                """,
                (
                    agent_run_id, m0_input_id, m0_input["bridge_id"], method, query_text, ticker,
                    m0_input["rewrite_type"], m0_input["m0_input_setting"], "no_entity_coverage",
                    "No entity coverage available; cannot generate candidate windows.", _dump(tool_trace),
                ),
            )
            conn.commit()
            return {"agent_run_id": agent_run_id, "decision_status": "no_entity_coverage"}
        for idx, candidate in enumerate(candidates):
            candidate_id = f"{agent_run_id}__cand_{idx:03d}"
            cur.execute(
                """
                INSERT INTO runtime.m0_candidate_windows (
                    candidate_window_id, m0_input_id, bridge_id, ticker, rewrite_type, m0_input_setting,
                    window_type, start_year, end_year, candidate_range, generation_method,
                    is_visible_to_model, uses_gold_label, rationale, priority_hint, relative_intent,
                    candidate_version, use_as_final_candidate
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (candidate_window_id) DO UPDATE SET
                    rationale = EXCLUDED.rationale,
                    priority_hint = EXCLUDED.priority_hint,
                    relative_intent = EXCLUDED.relative_intent,
                    candidate_version = EXCLUDED.candidate_version
                """,
                (
                    candidate_id, m0_input_id, m0_input["bridge_id"], ticker, m0_input["rewrite_type"],
                    m0_input["m0_input_setting"], candidate["window_type"], candidate["start_year"],
                    candidate["end_year"], _dump(candidate["candidate_range"]), candidate["generation_method"],
                    candidate["is_visible_to_model"], candidate["uses_gold_label"], candidate["rationale"],
                    candidate.get("priority_hint"), candidate.get("relative_intent"), candidate_version,
                    candidate.get("use_as_final_candidate", True),
                ),
            )
            metadata_probe = probe_content_metadata(conn, query_text, ticker, candidate["start_year"], candidate["end_year"])
            vector_probe = {"retrieved_chunk_ids": [], "retrieved_doc_ids": [], "retrieved_years": [], "matched_chunk_count": 0, "coverage_status": "skipped"}
            if method in {"vector_probe_agent_v1", "vector_probe_agent_v2"}:
                vector_probe = vector_probe_content(conn, query_text, ticker, candidate["start_year"], candidate["end_year"], top_k, allow_fallback)
                vector_probe["top_k"] = top_k
            probe_id = f"{candidate_id}__probe"
            retrieved_years = vector_probe.get("retrieved_years") or metadata_probe["years_with_evidence"]
            cur.execute(
                """
                INSERT INTO runtime.m0_evidence_probes (
                    probe_id, candidate_window_id, m0_input_id, query_text, ticker, start_year, end_year,
                    top_k, retrieved_chunk_ids, retrieved_doc_ids, retrieved_years, years_with_evidence,
                    missing_years, matched_chunk_count, matched_doc_count, evidence_density,
                    coverage_status, recommended_action, probe_method
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s)
                ON CONFLICT (probe_id) DO UPDATE SET coverage_status = EXCLUDED.coverage_status
                """,
                (
                    probe_id, candidate_id, m0_input_id, query_text, ticker, candidate["start_year"], candidate["end_year"],
                    top_k, _dump(vector_probe.get("retrieved_chunk_ids", [])), _dump(vector_probe.get("retrieved_doc_ids", [])),
                    _dump(retrieved_years), _dump(metadata_probe["years_with_evidence"]), _dump(metadata_probe["missing_years"]),
                    metadata_probe["matched_chunk_count"], metadata_probe["matched_doc_count"], _dump(metadata_probe["evidence_density"]),
                    metadata_probe["coverage_status"], metadata_probe["recommended_action"],
                    "metadata_plus_vector" if method in {"vector_probe_agent_v1", "vector_probe_agent_v2"} else "metadata_count",
                ),
            )
            score = score_candidate_window_v2(candidate, metadata_probe, vector_probe, diagnosis) if scoring_version == "v2" else score_candidate_window(candidate, metadata_probe, vector_probe, diagnosis)
            cur.execute(
                """
                UPDATE runtime.m0_candidate_windows
                SET score = %s, score_components = %s::jsonb
                WHERE candidate_window_id = %s
                """,
                (score["score"], _dump(score.get("score_components", {})), candidate_id),
            )
            item = {"candidate": {**candidate, "candidate_window_id": candidate_id}, "metadata_probe": metadata_probe, "vector_probe": vector_probe, **score}
            scored.append(item)
            tool_trace["candidates"].append(item)
        best = choose_best_window_v2(scored, diagnosis) if scoring_version == "v2" else choose_best_window(scored)
        selected = best.get("candidate", {})
        pred = selected.get("candidate_range")
        confidence = float(best.get("score", 0.0))
        decision_status = "predicted" if pred else "no_content_support"
        rationale = best.get("rationale", "No candidate selected.")
        cur.execute(
            """
            INSERT INTO runtime.m0_agent_runs (
                agent_run_id, m0_input_id, bridge_id, method, query_text, ticker, rewrite_type,
                m0_input_setting, predicted_start_year, predicted_end_year, predicted_range,
                selected_candidate_window_id, confidence, decision_status, rationale, tool_trace
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT (agent_run_id) DO UPDATE SET tool_trace = EXCLUDED.tool_trace
            """,
            (
                agent_run_id, m0_input_id, m0_input["bridge_id"], method, query_text, ticker,
                m0_input["rewrite_type"], m0_input["m0_input_setting"], pred[0] if pred else None,
                pred[1] if pred else None, _dump(pred), selected.get("candidate_window_id"), confidence,
                decision_status, rationale, _dump(tool_trace),
            ),
        )
        gold = get_gold_target(conn, m0_input["bridge_id"])
        metrics = compute_m0_metrics(pred, gold.get("gold_range"), gold.get("acceptable_ranges"))
        coverage_status = best.get("metadata_probe", {}).get("coverage_status", "unknown")
        can_run_m1 = best.get("metadata_probe", {}).get("matched_chunk_count", 0) > 0
        cur.execute(
            """
            INSERT INTO runtime.m0_agent_metrics (
                agent_run_id, gold_range, acceptable_ranges, overlap_score, start_boundary_error,
                end_boundary_error, mean_boundary_error, acceptable_accuracy, exact_match_accuracy,
                prediction_length_error, over_extension, under_extension, coverage_status, can_run_m1_retrieval
            ) VALUES (%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (agent_run_id) DO UPDATE SET
                overlap_score = EXCLUDED.overlap_score,
                mean_boundary_error = EXCLUDED.mean_boundary_error,
                acceptable_accuracy = EXCLUDED.acceptable_accuracy,
                exact_match_accuracy = EXCLUDED.exact_match_accuracy,
                coverage_status = EXCLUDED.coverage_status,
                can_run_m1_retrieval = EXCLUDED.can_run_m1_retrieval
            """,
            (
                agent_run_id, _dump(gold.get("gold_range")), _dump(gold.get("acceptable_ranges")),
                metrics["overlap_score"], metrics["start_boundary_error"], metrics["end_boundary_error"],
                metrics["mean_boundary_error"], metrics["acceptable_accuracy"], metrics["exact_match_accuracy"],
                metrics["prediction_length_error"], metrics["over_extension"], metrics["under_extension"],
                coverage_status, can_run_m1,
            ),
        )
        if store_oracle:
            store_oracle_analysis(conn, agent_run_id, m0_input, scored, selected, metrics, gold, method, candidate_version, scoring_version)
    conn.commit()
    return {
        "agent_run_id": agent_run_id,
        "decision_status": decision_status,
        "predicted_range": pred,
        "confidence": confidence,
        "metrics": metrics,
        "coverage_status": coverage_status,
        "selected_window_type": selected.get("window_type"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-input-id", required=True)
    parser.add_argument("--method", default="vector_probe_agent_v2", choices=["metadata_only_agent_v1", "vector_probe_agent_v1", "metadata_only_agent_v2", "vector_probe_agent_v2"])
    parser.add_argument("--agent-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--candidate-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--scoring-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--store-oracle-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--allow-fallback", action="store_true")
    add_connection_args(parser)
    args = parser.parse_args()
    with get_connection(args) as conn:
        result = run_agent(
            conn, args.m0_input_id, args.method, args.top_k, args.allow_fallback,
            args.agent_version, args.candidate_version, args.scoring_version,
            args.store_oracle_analysis,
        )
    print("M0 agent run summary")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
