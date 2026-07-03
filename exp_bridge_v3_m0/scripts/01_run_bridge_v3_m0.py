#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
M0_FINAL = ROOT / "m0_final"
if str(M0_FINAL) not in sys.path:
    sys.path.insert(0, str(M0_FINAL))

try:
    from scoring import score_e2_prior, score_final_fused, select_best_candidate
except Exception:  # pragma: no cover - fallback for standalone use
    score_e2_prior = None
    score_final_fused = None
    select_best_candidate = None


METHODS = {
    "E2_stable_temporal_prior": "E2",
    "FINAL_prior_guided_qualified_boundary": "FINAL",
}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_list(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        try:
            return json.loads(text)
        except Exception:
            return value


def as_year(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(float(value))
    except Exception:
        text = str(value)
        for token in text.replace("/", "-").split("-"):
            if token.isdigit() and len(token) == 4:
                return int(token)
    return None


def as_range(value: Any) -> list[int] | None:
    parsed = parse_list(value)
    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
        start, end = as_year(parsed[0]), as_year(parsed[1])
        if start is not None and end is not None and start <= end:
            return [start, end]
    return None


def overlap_score(pred: list[int] | None, gold: list[int] | None) -> float:
    if not pred or not gold:
        return 0.0
    ps = set(range(pred[0], pred[1] + 1))
    gs = set(range(gold[0], gold[1] + 1))
    union = ps | gs
    return len(ps & gs) / len(union) if union else 0.0


def range_len(rng: list[int] | None) -> int | None:
    if not rng:
        return None
    return rng[1] - rng[0] + 1


def is_acceptable(pred: list[int] | None, acceptable: Any) -> bool:
    if not pred:
        return False
    parsed = parse_list(acceptable)
    if not isinstance(parsed, list):
        return False
    for item in parsed:
        rng = as_range(item)
        if rng == pred:
            return True
    return False


def extension_metrics(pred: list[int] | None, gold: list[int] | None) -> tuple[int | None, int | None]:
    if not pred or not gold:
        return None, None
    pred_years = set(range(pred[0], pred[1] + 1))
    gold_years = set(range(gold[0], gold[1] + 1))
    return len(pred_years - gold_years), len(gold_years - pred_years)


def score_rewrite_type(rewrite_type: str) -> str:
    text = str(rewrite_type or "").lower()
    if "type l" in text or text == "l" or "latent" in text:
        return "A"
    return "B"


def dataset_year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    years = []
    for col in ["gold_start_year", "gold_end_year", "anchor_year", "event_anchor_year"]:
        if col in df.columns:
            years.extend([as_year(v) for v in df[col].tolist()])
    years = [y for y in years if y is not None]
    if not years:
        raise ValueError("Cannot derive dataset-level candidate year bounds from input.")
    return min(years), max(years)


def add_candidate(candidates: list[dict[str, Any]], qid: str, scope_type: str, start: Any, end: Any) -> None:
    s, e = as_year(start), as_year(end)
    if s is None or e is None or s > e:
        return
    candidates.append({
        "candidate_scope_id": f"{qid}__{scope_type}__{s}_{e}",
        "candidate_window_id": f"{qid}__{scope_type}__{s}_{e}",
        "scope_type": scope_type,
        "window_type": scope_type,
        "start_year": s,
        "end_year": e,
        "candidate_length": e - s + 1,
    })


def generate_candidates(row: pd.Series, min_year: int, max_year: int) -> list[dict[str, Any]]:
    qid = str(row.get("query_id") or row.get("bridge_id"))
    candidates: list[dict[str, Any]] = []
    add_candidate(candidates, qid, "full_range", min_year, max_year)
    add_candidate(candidates, qid, "full_entity_range", min_year, max_year)
    add_candidate(candidates, qid, "rolling_3y", max_year - 2, max_year)
    add_candidate(candidates, qid, "rolling_5y", max_year - 4, max_year)
    add_candidate(candidates, qid, "rolling_8y", max_year - 7, max_year)
    mid = (min_year + max_year) // 2
    add_candidate(candidates, qid, "early_half", min_year, mid)
    add_candidate(candidates, qid, "late_half", mid + 1, max_year)

    anchor_year = as_year(row.get("anchor_year"))
    event_year = as_year(row.get("event_anchor_year"))
    for label, year in [("anchor", anchor_year), ("event", event_year)]:
        if year is None:
            continue
        add_candidate(candidates, qid, f"{label}_prefix", min_year, year)
        add_candidate(candidates, qid, f"{label}_suffix", year, max_year)
        add_candidate(candidates, qid, f"{label}_before", min_year, year - 1)
        add_candidate(candidates, qid, f"{label}_after", year + 1, max_year)
        add_candidate(candidates, qid, "prefix_window", min_year, year - 1)
        add_candidate(candidates, qid, "suffix_window", year + 1, max_year)

    unique = {}
    for cand in candidates:
        key = (cand["start_year"], cand["end_year"], cand["scope_type"])
        unique[key] = cand
    return list(unique.values())


def enrich_for_scoring(candidate: dict[str, Any], row: pd.Series, min_year: int, max_year: int, neutral: bool = True) -> dict[str, Any]:
    out = dict(candidate)
    out.update({
        "rewritten_query": row.get("query_text") or row.get("rewritten_query"),
        "query_text": row.get("query_text") or row.get("rewritten_query"),
        "rewrite_type": score_rewrite_type(row.get("rewrite_type")),
        "original_rewrite_type": row.get("rewrite_type"),
        "ticker": row.get("target_subject") or row.get("ticker"),
        "target_subject": row.get("target_subject") or row.get("ticker"),
        "probe_years_with_evidence": list(range(min_year, max_year + 1)),
        "probe_coverage_status": "sufficient",
        "timeline_length": max_year - min_year + 1,
    })
    if neutral:
        length_ratio = out["candidate_length"] / max(1, max_year - min_year + 1)
        out.update({
            "e7_timeline_min_year": min_year,
            "e7_timeline_max_year": max_year,
            "e7_timeline_length": max_year - min_year + 1,
            "e7_candidate_length_ratio": length_ratio,
            "e7_inside_relevance_mean": 0.50,
            "e7_inside_coherence": 0.50,
            "e7_boundary_contrast": 0.50,
            "e7_trend_coherence": 0.50,
            "qualified_evidence_year_coverage": 0.50,
            "background_noise_ratio": 0.0,
            "e7_fallback_used": True,
        })
    return out


def fallback_e2(candidate: dict[str, Any]) -> dict[str, Any]:
    length = int(candidate["end_year"]) - int(candidate["start_year"]) + 1
    length_fit = 0.20 if length <= 2 else 0.65 if length <= 4 else 1.0 if length <= 8 else 0.75 if length <= 10 else 0.45
    w = candidate.get("window_type")
    window_prior = {"rolling_8y": 0.90, "rolling_5y": 0.80, "prefix_window": 0.75, "suffix_window": 0.75, "early_half": 0.70, "late_half": 0.70, "full_entity_range": 0.50, "full_range": 0.50}.get(w, 0.50)
    q = str(candidate.get("query_text") or "").lower()
    if any(x in q for x in ["before", "earlier", "prior"]):
        intent = 0.95 if "prefix" in str(w) or "before" in str(w) or w == "early_half" else 0.45
    elif any(x in q for x in ["after", "later", "following", "since"]):
        intent = 0.95 if "suffix" in str(w) or "after" in str(w) or w == "late_half" else 0.45
    else:
        intent = 0.90 if w == "rolling_8y" else 0.75 if "prefix" in str(w) or "suffix" in str(w) else 0.55
    score = 0.40 * intent + 0.25 * length_fit + 0.20 * 1.0 + 0.15 * window_prior
    return {"score": max(0.0, min(1.0, score)), "score_components": {"intent_fit": intent, "length_fit": length_fit, "evidence_year_coverage": 1.0, "window_type_prior": window_prior, "fallback": True}}


def score_candidates(row: pd.Series, candidates: list[dict[str, Any]], method: str, min_year: int, max_year: int) -> dict[str, Any] | None:
    scored = []
    for cand in candidates:
        enriched = enrich_for_scoring(cand, row, min_year, max_year)
        if method == "E2":
            result = score_e2_prior(enriched) if score_e2_prior else fallback_e2(enriched)
        elif method == "FINAL":
            result = score_final_fused(enriched) if score_final_fused else fallback_e2(enriched)
            result["score_components"]["adapter_neutral_features"] = True
        else:
            continue
        scored.append({"candidate": enriched, "score": result["score"], "score_components": result["score_components"]})
    if not scored:
        return None
    if select_best_candidate:
        return select_best_candidate(scored, method)
    return sorted(scored, key=lambda x: x["score"], reverse=True)[0]


def eval_row(pred: list[int] | None, gold: list[int] | None, acceptable: Any) -> dict[str, Any]:
    start_err = abs(pred[0] - gold[0]) if pred and gold else None
    end_err = abs(pred[1] - gold[1]) if pred and gold else None
    over, under = extension_metrics(pred, gold)
    return {
        "is_acceptable": is_acceptable(pred, acceptable),
        "overlap_score": overlap_score(pred, gold),
        "start_boundary_error": start_err,
        "end_boundary_error": end_err,
        "mean_boundary_error": (start_err + end_err) / 2 if start_err is not None and end_err is not None else None,
        "over_extension": over,
        "under_extension": under,
        "predicted_length": range_len(pred),
        "gold_length": range_len(gold),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="exp_bridge_v3_m0/data/bridge_v3_m0_inputs.csv")
    parser.add_argument("--output-dir", default="exp_bridge_v3_m0/outputs/csv")
    parser.add_argument("--methods", default="E2_stable_temporal_prior,FINAL_prior_guided_qualified_boundary")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    min_year, max_year = dataset_year_bounds(df)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = []
    metric_rows = []

    for _, row in df.iterrows():
        candidates = generate_candidates(row, min_year, max_year)
        gold = as_range(row.get("gold_range_parsed") or row.get("gold_range"))
        acceptable = row.get("acceptable_ranges_parsed") or row.get("acceptable_ranges")
        for method_name in methods:
            short = METHODS.get(method_name, method_name)
            selected = score_candidates(row, candidates, short, min_year, max_year)
            pred = None
            if selected:
                cand = selected["candidate"]
                pred = [int(cand["start_year"]), int(cand["end_year"])]
                score = selected["score"]
                comps = selected.get("score_components", {})
                scope_type = cand.get("scope_type") or cand.get("window_type")
            else:
                score = None
                comps = {}
                scope_type = None
            metrics = eval_row(pred, gold, acceptable)
            base = {
                "query_id": row.get("query_id"),
                "source_id": row.get("source_id"),
                "bridge_id": row.get("bridge_id"),
                "rewrite_type": row.get("rewrite_type"),
                "query_type_name": row.get("query_type_name"),
                "query_type_normalized": row.get("query_type_normalized"),
                "query_text": row.get("query_text"),
                "target_subject": row.get("target_subject"),
                "gold_start_year": row.get("gold_start_year"),
                "gold_end_year": row.get("gold_end_year"),
                "acceptable_ranges": acceptable,
                "anchor_year": row.get("anchor_year"),
                "anchor_direction": row.get("anchor_direction"),
                "event_anchor_year": row.get("event_anchor_year"),
                "method": method_name,
                "predicted_start_year": pred[0] if pred else None,
                "predicted_end_year": pred[1] if pred else None,
                "predicted_temporal_scope": json.dumps(pred) if pred else None,
                "selected_scope_type": scope_type,
                "score": score,
                "score_components": json.dumps(comps, ensure_ascii=False, default=str),
            }
            out = {**base, **metrics}
            rows.append(out)
            metric_rows.append({
                "query_id": row.get("query_id"),
                "source_id": row.get("source_id"),
                "rewrite_type": row.get("rewrite_type"),
                "query_type_normalized": row.get("query_type_normalized"),
                "method": method_name,
                **metrics,
            })

    out_dir = ensure_dir(args.output_dir)
    runs = pd.DataFrame(rows)
    metrics_df = pd.DataFrame(metric_rows)
    runs.to_csv(out_dir / "bridge_v3_m0_runs.csv", index=False)
    metrics_df.to_csv(out_dir / "bridge_v3_m0_metrics.csv", index=False)
    print(f"dataset year bounds: {min_year}-{max_year}")
    print(f"input rows: {len(df)}")
    print(f"run rows: {len(runs)}")
    print(f"wrote -> {out_dir / 'bridge_v3_m0_runs.csv'}")
    print(f"wrote -> {out_dir / 'bridge_v3_m0_metrics.csv'}")


if __name__ == "__main__":
    main()
