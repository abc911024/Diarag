#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from db_config import add_connection_args, get_connection, quote_ident


def read_jsonl(path: Path, optional: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        level = "Warning" if optional else "Warning"
        print(f"{level}: missing file {path}; skipping")
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        print(f"Warning: optional file {path} missing or empty; skipping")
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_jsonish(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (list, dict)):
        return value
    return json.loads(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/qa"))
    parser.add_argument("--schema", default="qa")
    add_connection_args(parser)
    args = parser.parse_args()
    schema = quote_ident(args.schema)

    raw = read_jsonl(args.data_dir / "raw_qa_items.jsonl")
    candidates = read_jsonl(args.data_dir / "bridge_candidates.jsonl") + read_jsonl(args.data_dir / "rejected_candidates.jsonl", optional=True)
    records = read_jsonl(args.data_dir / "bridge_rewrite_dataset_v3_qa.jsonl")
    targets = read_jsonl(args.data_dir / "bridge_m0_eval_targets_qa.jsonl")
    coverage = read_jsonl(args.data_dir / "entity_coverage.jsonl", optional=True)
    inputs = read_jsonl(args.data_dir / "bridge_m0_inputs_qa.jsonl")
    baselines = read_csv(args.data_dir / "m0_baseline_results.csv")

    counts = {name: 0 for name in [
        "raw_qa_items", "bridge_candidates", "bridge_records", "temporal_annotations",
        "entity_coverage", "m0_inputs", "m0_predictions", "m0_metrics",
    ]}

    with get_connection(args) as conn:
        with conn.cursor() as cur:
            for r in raw:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.raw_qa_items (
                        source_id, source_dataset, raw_json, ticker, stock_name, original_question,
                        temporal_context_label, query_type, query_type_id, generation_score, correct_answer_key
                    ) VALUES (%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (source_id) DO UPDATE SET
                        raw_json = EXCLUDED.raw_json,
                        ticker = EXCLUDED.ticker,
                        stock_name = EXCLUDED.stock_name,
                        original_question = EXCLUDED.original_question,
                        temporal_context_label = EXCLUDED.temporal_context_label,
                        query_type = EXCLUDED.query_type,
                        query_type_id = EXCLUDED.query_type_id,
                        generation_score = EXCLUDED.generation_score,
                        correct_answer_key = EXCLUDED.correct_answer_key
                    """,
                    (
                        r["source_id"], r.get("source_dataset", "DQABench"), dumps(r.get("raw_json")),
                        r.get("ticker"), r.get("stock_name"), r.get("original_question"),
                        r.get("temporal_context_label"), r.get("query_type"), r.get("query_type_id"),
                        r.get("generation_score"), r.get("correct_answer_key"),
                    ),
                )
                counts["raw_qa_items"] += 1
            for r in candidates:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.bridge_candidates (
                        source_id, is_bridge_candidate, candidate_reason, reject_reason, duration_days,
                        detected_diachronic_intent, detected_factoid
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (source_id) DO UPDATE SET
                        is_bridge_candidate = EXCLUDED.is_bridge_candidate,
                        candidate_reason = EXCLUDED.candidate_reason,
                        reject_reason = EXCLUDED.reject_reason,
                        duration_days = EXCLUDED.duration_days,
                        detected_diachronic_intent = EXCLUDED.detected_diachronic_intent,
                        detected_factoid = EXCLUDED.detected_factoid
                    """,
                    (
                        r["source_id"], r["is_bridge_candidate"], r.get("candidate_reason"),
                        r.get("reject_reason"), r.get("duration_days"),
                        r.get("detected_diachronic_intent"), r.get("detected_factoid"),
                    ),
                )
                counts["bridge_candidates"] += 1
            for r in records:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.bridge_records (
                        bridge_id, source_id, source_dataset, original_query, pseudo_query, rewritten_query,
                        rewrite_type, rewrite_status, rewrite_reason, rewrite_quality_status,
                        rewrite_quality_flags, ticker, stock_name, metadata_used_for_rewrite,
                        annotation_uses_metadata
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
                    ON CONFLICT (bridge_id) DO UPDATE SET
                        rewritten_query = EXCLUDED.rewritten_query,
                        rewrite_status = EXCLUDED.rewrite_status,
                        rewrite_reason = EXCLUDED.rewrite_reason,
                        rewrite_quality_status = EXCLUDED.rewrite_quality_status,
                        rewrite_quality_flags = EXCLUDED.rewrite_quality_flags
                    """,
                    (
                        r["bridge_id"], r["source_id"], r.get("source_dataset", "DQABench"),
                        r.get("original_query"), r.get("pseudo_query"), r.get("rewritten_query"),
                        r.get("rewrite_type"), r.get("rewrite_status"), r.get("rewrite_reason"),
                        r.get("rewrite_quality_status"), dumps(r.get("rewrite_quality_flags", {})),
                        r.get("ticker"), r.get("stock_name"), r.get("metadata_used_for_rewrite", False),
                        r.get("annotation_uses_metadata", True),
                    ),
                )
                counts["bridge_records"] += 1
            for r in targets:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.temporal_annotations (
                        bridge_id, gold_start_date, gold_end_date, gold_start_year, gold_end_year,
                        gold_range, acceptable_ranges, temporal_scope_type, source_temporal_label,
                        anchor_year, anchor_direction, annotation_source
                    ) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s)
                    ON CONFLICT (bridge_id) DO UPDATE SET
                        gold_start_date = EXCLUDED.gold_start_date,
                        gold_end_date = EXCLUDED.gold_end_date,
                        gold_start_year = EXCLUDED.gold_start_year,
                        gold_end_year = EXCLUDED.gold_end_year,
                        gold_range = EXCLUDED.gold_range,
                        acceptable_ranges = EXCLUDED.acceptable_ranges,
                        temporal_scope_type = EXCLUDED.temporal_scope_type
                    """,
                    (
                        r["bridge_id"], r.get("gold_start_date"), r.get("gold_end_date"),
                        r.get("gold_start_year"), r.get("gold_end_year"), dumps(r.get("gold_range")),
                        dumps(r.get("acceptable_ranges")), r.get("temporal_scope_type"),
                        r.get("source_temporal_label"), r.get("anchor_year"),
                        r.get("anchor_direction"), r.get("annotation_source", "plot_time_bounds"),
                    ),
                )
                counts["temporal_annotations"] += 1
            for r in coverage:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.entity_coverage (
                        entity_id, ticker, stock_name, min_year, max_year, available_years, coverage_source
                    ) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s)
                    ON CONFLICT (entity_id) DO UPDATE SET
                        ticker = EXCLUDED.ticker,
                        stock_name = EXCLUDED.stock_name,
                        min_year = EXCLUDED.min_year,
                        max_year = EXCLUDED.max_year,
                        available_years = EXCLUDED.available_years,
                        coverage_source = EXCLUDED.coverage_source
                    """,
                    (
                        r["entity_id"], r.get("ticker"), r.get("stock_name"), r.get("min_year"),
                        r.get("max_year"), dumps(r.get("available_years", [])),
                        r.get("coverage_source", "derived_from_qa_gold_coverage_for_m0_experiment"),
                    ),
                )
                counts["entity_coverage"] += 1
            for r in inputs:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.m0_inputs (
                        m0_input_id, bridge_id, m0_input_setting, m0_input_json, rewrite_type,
                        temporal_need_type, evidence_policy, input_contains_gold_answer, input_difficulty_note
                    ) VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s)
                    ON CONFLICT (m0_input_id) DO UPDATE SET
                        m0_input_json = EXCLUDED.m0_input_json,
                        input_contains_gold_answer = EXCLUDED.input_contains_gold_answer,
                        input_difficulty_note = EXCLUDED.input_difficulty_note
                    """,
                    (
                        r["m0_input_id"], r["bridge_id"], r["m0_input_setting"],
                        dumps(r["m0_input_json"]), r.get("rewrite_type"),
                        r.get("temporal_need_type", "historical_trend"),
                        r.get("evidence_policy", "broad_historical_coverage"),
                        r.get("input_contains_gold_answer", False), r.get("input_difficulty_note"),
                    ),
                )
                counts["m0_inputs"] += 1
            for r in baselines:
                prediction_id = r.get("prediction_id")
                m0_input_id = r.get("m0_input_id")
                method = r.get("method")
                pred = parse_jsonish(r.get("predicted_range"))
                if not prediction_id or not m0_input_id or not method or not pred:
                    print(f"Warning: skipping incomplete baseline row: {r}")
                    continue
                cur.execute(
                    f"""
                    INSERT INTO {schema}.m0_predictions (
                        prediction_id, m0_input_id, method, predicted_start_year, predicted_end_year,
                        predicted_range, rationale
                    ) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s)
                    ON CONFLICT (prediction_id) DO UPDATE SET
                        predicted_start_year = EXCLUDED.predicted_start_year,
                        predicted_end_year = EXCLUDED.predicted_end_year,
                        predicted_range = EXCLUDED.predicted_range,
                        rationale = EXCLUDED.rationale
                    """,
                    (
                        prediction_id, m0_input_id, method, int(pred[0]), int(pred[1]),
                        dumps(pred), r.get("rationale"),
                    ),
                )
                counts["m0_predictions"] += 1
                if r.get("overlap_score") not in (None, ""):
                    cur.execute(
                        f"""
                        INSERT INTO {schema}.m0_metrics (
                            prediction_id, overlap_score, start_boundary_error, end_boundary_error,
                            mean_boundary_error, acceptable_accuracy, exact_match_accuracy,
                            prediction_length_error, over_extension, under_extension
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (prediction_id) DO UPDATE SET
                            overlap_score = EXCLUDED.overlap_score,
                            start_boundary_error = EXCLUDED.start_boundary_error,
                            end_boundary_error = EXCLUDED.end_boundary_error,
                            mean_boundary_error = EXCLUDED.mean_boundary_error,
                            acceptable_accuracy = EXCLUDED.acceptable_accuracy,
                            exact_match_accuracy = EXCLUDED.exact_match_accuracy,
                            prediction_length_error = EXCLUDED.prediction_length_error,
                            over_extension = EXCLUDED.over_extension,
                            under_extension = EXCLUDED.under_extension
                        """,
                        (
                            prediction_id, parse_float(r.get("overlap_score")),
                            parse_int(r.get("start_boundary_error")), parse_int(r.get("end_boundary_error")),
                            parse_float(r.get("mean_boundary_error")), parse_bool(r.get("acceptable_accuracy")),
                            parse_bool(r.get("exact_match_accuracy")), parse_int(r.get("prediction_length_error")),
                            parse_int(r.get("over_extension")), parse_int(r.get("under_extension")),
                        ),
                    )
                    counts["m0_metrics"] += 1
        conn.commit()

    for table, count in counts.items():
        print(f"{args.schema}.{table}: upserted {count}")


if __name__ == "__main__":
    main()
