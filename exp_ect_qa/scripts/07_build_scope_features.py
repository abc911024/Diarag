#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "query_id",
    "candidate_scope_id",
    "analysis_focus",
    "start_time_unit",
    "end_time_unit",
    "scope_type",
    "intent_fit",
    "length_fit",
    "scope_type_prior",
    "evidence_time_coverage",
    "candidate_length_ratio",
    "long_window_penalty",
    "directional_adjustment",
    "qualified_inside_relevance",
    "qualified_evidence_time_coverage",
    "qualified_boundary_contrast",
    "qualified_trend_coherence",
    "background_noise_ratio",
    "feature_quality_flags_json",
]


def length_fit(length: int) -> float:
    if length <= 2:
        return 0.20
    if 3 <= length <= 4:
        return 0.65
    if 5 <= length <= 8:
        return 1.00
    if 9 <= length <= 10:
        return 0.75
    return 0.45


def scope_type_prior(scope_type: str) -> float:
    return {
        "full_available_scope": 0.50,
        "single_time_unit": 0.20,
        "rolling_3": 0.60,
        "rolling_5": 0.80,
        "rolling_8": 0.90,
        "recent_3": 0.60,
        "recent_5": 0.65,
        "content_supported_scope": 0.20,
    }.get(scope_type, 0.50)


def intent_fit(temporal_intent: str, scope_type: str, length: int) -> float:
    if temporal_intent == "diachronic_trend":
        return 0.95 if scope_type in {"rolling_5", "rolling_8", "full_available_scope"} else 0.60
    if temporal_intent == "explicit_time":
        return 0.75 if length <= 3 else 0.55
    if temporal_intent == "relative_time":
        return 0.85 if scope_type in {"recent_3", "recent_5"} else 0.60
    if temporal_intent == "comparison":
        return 0.80 if length >= 2 else 0.40
    return 0.55


def long_window_penalty(length_ratio: float, scope_type: str) -> float:
    penalty = 0.0
    if length_ratio > 0.70:
        penalty = 0.05
    if length_ratio > 0.90:
        penalty = 0.15
    if scope_type == "full_available_scope":
        penalty += 0.05
    return min(0.20, penalty)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-inputs", default="exp_ect_qa/processed/ectqa_m0_inputs.csv")
    parser.add_argument("--evidence-units", default="exp_ect_qa/processed/ectqa_evidence_units.csv")
    parser.add_argument("--candidate-scopes", default="exp_ect_qa/processed/ectqa_candidate_temporal_scopes.csv")
    parser.add_argument("--output", default="exp_ect_qa/processed/ectqa_candidate_scope_features.csv")
    args = parser.parse_args()

    inputs = pd.read_csv(args.m0_inputs)
    ev = pd.read_csv(args.evidence_units)
    try:
        scopes = pd.read_csv(args.candidate_scopes)
    except pd.errors.EmptyDataError:
        scopes = pd.DataFrame(columns=["query_id"])
    if scopes.empty:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(out, index=False)
        print(f"WARNING: no candidate temporal scopes found; wrote empty features -> {out}")
        return
    input_by_id = inputs.set_index("query_id").to_dict("index")
    rows = []
    for _, scope in scopes.iterrows():
        query_id = scope["query_id"]
        qev = ev[(ev["query_id"] == query_id) & (ev["usable_for_m0"].astype(str).str.lower().isin(["true", "1"]))]
        years = sorted({int(str(x)[:4]) for x in qev["time_unit"].dropna() if str(x)[:4].isdigit()})
        start, end = int(scope["start_time_unit"]), int(scope["end_time_unit"])
        inside = [y for y in years if start <= y <= end]
        length = end - start + 1
        timeline_len = max(1, max(years) - min(years) + 1) if years else length
        coverage = len(set(inside)) / max(1, length)
        length_ratio = length / timeline_len
        m0 = input_by_id.get(query_id, {})
        temporal_intent = m0.get("temporal_intent", "unknown")
        flags = {
            "qualification_available": False,
            "uses_gold_temporal_scope": False,
            "feature_basis": "raw_time_indexed_evidence_proxy",
        }
        row = {
            "query_id": query_id,
            "candidate_scope_id": scope["candidate_scope_id"],
            "analysis_focus": scope.get("analysis_focus"),
            "start_time_unit": start,
            "end_time_unit": end,
            "scope_type": scope["scope_type"],
            "intent_fit": intent_fit(temporal_intent, scope["scope_type"], length),
            "length_fit": length_fit(length),
            "scope_type_prior": scope_type_prior(scope["scope_type"]),
            "evidence_time_coverage": coverage,
            "candidate_length_ratio": length_ratio,
            "long_window_penalty": long_window_penalty(length_ratio, scope["scope_type"]),
            "directional_adjustment": 0.0,
            "qualified_inside_relevance": coverage,
            "qualified_evidence_time_coverage": coverage,
            "qualified_boundary_contrast": 0.0,
            "qualified_trend_coherence": 0.0,
            "background_noise_ratio": 0.0,
            "feature_quality_flags_json": json.dumps(flags),
            # Backward-compatible aliases for m0_final scoring functions.
            "target_subject": scope.get("analysis_focus"),
            "start_year": start,
            "end_year": end,
            "window_type": scope["scope_type"],
            "evidence_year_coverage": coverage,
            "qualified_evidence_year_coverage": coverage,
            "candidate_window_id": scope["candidate_scope_id"],
            "m0_input_id": query_id,
            "bridge_id": query_id,
            "query_text": m0.get("query_text"),
            "rewritten_query": m0.get("query_text"),
            "rewrite_type": "A",
            "gold_range": m0.get("gold_temporal_scope"),
            "acceptable_ranges": m0.get("acceptable_temporal_scopes"),
            "e7_inside_relevance_mean": coverage,
            "e7_inside_coherence": coverage,
            "e7_boundary_contrast": 0.0,
            "e7_trend_coherence": 0.0,
            "e7_candidate_length_ratio": length_ratio,
            "e7_long_window_penalty": long_window_penalty(length_ratio, scope["scope_type"]),
            "raw_inside_relevance_mean": coverage,
            "raw_inside_coherence": coverage,
            "raw_boundary_contrast": 0.0,
            "raw_trend_coherence": 0.0,
            "raw_candidate_length_ratio": length_ratio,
            "raw_long_window_penalty": long_window_penalty(length_ratio, scope["scope_type"]),
        }
        rows.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
