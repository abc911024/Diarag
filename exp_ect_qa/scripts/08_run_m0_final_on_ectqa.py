#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "m0_final"))

from evaluator import as_range, evaluate_prediction  # noqa: E402
from scoring import (  # noqa: E402
    score_e2_prior,
    score_e6b_raw_boundary,
    score_e7_qualified_boundary,
    score_final_fused,
    select_best_candidate,
)


SCORERS = {
    "E2": score_e2_prior,
    "E6b": score_e6b_raw_boundary,
    "E7": score_e7_qualified_boundary,
    "FINAL": score_final_fused,
}


def clean_record(record: dict) -> dict:
    cleaned = {}
    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    if cleaned.get("query_text") is None:
        cleaned["query_text"] = ""
    if cleaned.get("rewritten_query") is None:
        cleaned["rewritten_query"] = cleaned.get("query_text") or ""
    if cleaned.get("rewrite_type") is None:
        cleaned["rewrite_type"] = "A"
    return cleaned


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def has_gold(value) -> bool:
    return as_range(value) is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="exp_ect_qa/processed/ectqa_candidate_scope_features.csv")
    parser.add_argument("--outputs-dir", default="exp_ect_qa/outputs")
    parser.add_argument("--reports-dir", default="exp_ect_qa/reports")
    parser.add_argument("--experiments", default="E2,E6b,E7,FINAL")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    runs = []
    metrics = []
    skipped = []
    for exp in experiments:
        scorer = SCORERS.get(exp)
        if scorer is None:
            skipped.append(f"{exp}: unknown experiment")
            continue
        for query_id, group in features.groupby("query_id"):
            candidates = [clean_record(c) for c in group.to_dict("records")]
            scored = [{"candidate": c, **scorer(c)} for c in candidates]
            selected = select_best_candidate(scored, exp)
            if not selected:
                continue
            cand = selected["candidate"]
            pred = [int(cand["start_year"]), int(cand["end_year"])]
            gold = cand.get("gold_range")
            acceptable = cand.get("acceptable_ranges")
            run_id = f"{exp}__{query_id}"
            runs.append({
                "run_id": run_id,
                "experiment": exp,
                "query_id": query_id,
                "candidate_scope_id": cand.get("candidate_scope_id"),
                "analysis_focus": cand.get("analysis_focus"),
                "predicted_temporal_scope": json.dumps(pred),
                "score": selected["score"],
                "score_components_json": json.dumps(selected["score_components"], ensure_ascii=False),
            })
            if has_gold(gold):
                metrics.append({"run_id": run_id, "experiment": exp, "query_id": query_id, **evaluate_prediction(pred, gold, acceptable)})
            else:
                metrics.append({
                    "run_id": run_id,
                    "experiment": exp,
                    "query_id": query_id,
                    "acceptable_accuracy": None,
                    "overlap_score": None,
                    "start_boundary_error": None,
                    "end_boundary_error": None,
                    "mean_boundary_error": None,
                    "predicted_length": pred[1] - pred[0] + 1,
                    "over_extension": None,
                    "under_extension": None,
                })

    outputs = Path(args.outputs_dir)
    reports = Path(args.reports_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    runs_df = pd.DataFrame(runs)
    metrics_df = pd.DataFrame(metrics)
    runs_df.to_csv(outputs / "ectqa_m0_runs.csv", index=False)
    metrics_df.to_csv(outputs / "ectqa_m0_metrics.csv", index=False)
    if not metrics_df.empty and metrics_df["acceptable_accuracy"].notna().any():
        summary = metrics_df.groupby("experiment", dropna=False).agg(
            acceptable_accuracy=("acceptable_accuracy", "mean"),
            avg_overlap=("overlap_score", "mean"),
            avg_boundary_error=("mean_boundary_error", "mean"),
            avg_predicted_length=("predicted_length", "mean"),
        ).reset_index()
    else:
        summary = runs_df.groupby("experiment").agg(
            n=("run_id", "count"),
            avg_score=("score", "mean"),
        ).reset_index() if not runs_df.empty else pd.DataFrame()
    summary.to_csv(outputs / "ectqa_m0_method_comparison.csv", index=False)

    lines = [
        "# ECT-QA M0 Experiment Report",
        "",
        "This report reuses m0_final scoring functions on processed ECT-QA framework CSVs. It does not modify m0_final scoring logic.",
        "",
        "## Evaluation",
        "",
    ]
    if metrics_df.empty or not metrics_df["acceptable_accuracy"].notna().any():
        lines.append("No reliable gold temporal scope was available, so M0 acceptable accuracy was not computed. Report predicted scope distributions and consider downstream answer validation.")
    else:
        lines.append("Gold temporal scope was available for at least some rows, so temporal scope metrics were computed.")
    if skipped:
        lines += ["", "## Skipped", "", *[f"- {x}" for x in skipped]]
    lines += ["", "## Summary", "", dataframe_to_markdown(summary)]
    (reports / "ectqa_m0_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote outputs -> {outputs}")
    print(f"wrote report -> {reports / 'ectqa_m0_experiment_report.md'}")


if __name__ == "__main__":
    main()
