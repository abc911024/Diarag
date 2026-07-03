#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("query_id", "count"),
            acceptable_accuracy=("is_acceptable", "mean"),
            avg_overlap=("overlap_score", "mean"),
            avg_start_error=("start_boundary_error", "mean"),
            avg_end_error=("end_boundary_error", "mean"),
            avg_boundary_error=("mean_boundary_error", "mean"),
            avg_over_extension=("over_extension", "mean"),
            avg_under_extension=("under_extension", "mean"),
            avg_predicted_length=("predicted_length", "mean"),
            avg_gold_length=("gold_length", "mean"),
        )
        .reset_index()
    )


def classify_error(row: pd.Series) -> str:
    overlap = float(row.get("overlap_score") or 0)
    pred_start = row.get("predicted_start_year")
    pred_end = row.get("predicted_end_year")
    gold_start = row.get("gold_start_year")
    gold_end = row.get("gold_end_year")
    anchor_year = row.get("anchor_year")
    anchor_direction = str(row.get("anchor_direction") or "").lower()

    if overlap == 0:
        return "no_overlap"
    if pd.notna(pred_end) and pd.notna(gold_start) and pred_end < gold_start:
        return "shifted_before"
    if pd.notna(pred_start) and pd.notna(gold_end) and pred_start > gold_end:
        return "shifted_after"
    if pd.notna(anchor_year):
        if anchor_direction == "before" and pd.notna(pred_start) and pred_start >= anchor_year:
            return "wrong_anchor_direction"
        if anchor_direction == "after" and pd.notna(pred_end) and pred_end <= anchor_year:
            return "wrong_anchor_direction"
    over = row.get("over_extension")
    under = row.get("under_extension")
    if pd.notna(over) and pd.notna(under):
        if over > under:
            return "over_extended"
        if under > over:
            return "under_extended"
    return "other"


def load_inputs(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def conservative_findings(summary_method: pd.DataFrame, summary_type: pd.DataFrame, errors: pd.DataFrame) -> list[str]:
    findings = []
    if not summary_method.empty:
        best = summary_method.sort_values("acceptable_accuracy", ascending=False).iloc[0]
        worst = summary_method.sort_values("acceptable_accuracy", ascending=True).iloc[0]
        findings.append(
            f"By acceptable accuracy, the strongest method in this adapter run is `{best['method']}` "
            f"({best['acceptable_accuracy']:.4f}); the lowest is `{worst['method']}` ({worst['acceptable_accuracy']:.4f})."
        )
    if not summary_type.empty:
        type_avg = summary_type.groupby("query_type_normalized")["acceptable_accuracy"].mean().reset_index()
        hardest = type_avg.sort_values("acceptable_accuracy", ascending=True).iloc[0]
        findings.append(
            f"Averaged across methods, `{hardest['query_type_normalized']}` has the lowest acceptable accuracy "
            f"({hardest['acceptable_accuracy']:.4f}) and is the hardest query type in this run."
        )
    if not errors.empty:
        counts = errors["error_type"].value_counts()
        top = counts.index[0]
        findings.append(f"The most frequent error type is `{top}` with {int(counts.iloc[0])} cases.")
        over_mean = errors["over_extension"].mean()
        under_mean = errors["under_extension"].mean()
        if pd.notna(over_mean) and pd.notna(under_mean):
            if over_mean > under_mean:
                findings.append("Among error cases, over-extension is larger on average than under-extension.")
            elif under_mean > over_mean:
                findings.append("Among error cases, under-extension is larger on average than over-extension.")
            else:
                findings.append("Among error cases, average over-extension and under-extension are similar.")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="exp_bridge_v3_m0/outputs/csv/bridge_v3_m0_runs.csv")
    parser.add_argument("--input", default="exp_bridge_v3_m0/data/bridge_v3_m0_inputs.csv")
    parser.add_argument("--output-dir", default="exp_bridge_v3_m0/outputs")
    args = parser.parse_args()

    runs_path = Path(args.runs)
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs CSV not found: {runs_path}")

    runs = pd.read_csv(runs_path)
    out_root = ensure_dir(args.output_dir)
    csv_dir = ensure_dir(out_root / "csv")
    md_dir = ensure_dir(out_root / "markdown")

    summary_method = summarize(runs, ["method"])
    summary_type = summarize(runs, ["method", "query_type_normalized"])
    errors = runs[runs["is_acceptable"] == False].copy()
    errors["error_type"] = errors.apply(classify_error, axis=1)

    summary_method.to_csv(csv_dir / "bridge_v3_m0_summary_by_method.csv", index=False)
    summary_type.to_csv(csv_dir / "bridge_v3_m0_summary_by_type.csv", index=False)
    errors.to_csv(csv_dir / "bridge_v3_m0_error_cases.csv", index=False)

    metrics_cols = [
        "query_id", "source_id", "rewrite_type", "query_type_normalized", "method",
        "is_acceptable", "overlap_score", "start_boundary_error", "end_boundary_error",
        "mean_boundary_error", "over_extension", "under_extension",
        "predicted_length", "gold_length",
    ]
    runs[[c for c in metrics_cols if c in runs.columns]].to_csv(csv_dir / "bridge_v3_m0_metrics.csv", index=False)

    inputs = load_inputs(Path(args.input))
    total_rows = len(inputs) if not inputs.empty else runs["query_id"].nunique()
    source_count = inputs["source_id"].nunique() if not inputs.empty and "source_id" in inputs else runs["source_id"].nunique()
    rewrite_dist = inputs["rewrite_type"].value_counts(dropna=False).reset_index() if not inputs.empty else runs["rewrite_type"].value_counts(dropna=False).reset_index()
    rewrite_dist.columns = ["rewrite_type", "n"]
    qtype_dist = inputs["query_type_normalized"].value_counts(dropna=False).reset_index() if not inputs.empty else runs["query_type_normalized"].value_counts(dropna=False).reset_index()
    qtype_dist.columns = ["query_type_normalized", "n"]
    temporal_dist = inputs["temporal_scope_type"].value_counts(dropna=False).reset_index() if not inputs.empty and "temporal_scope_type" in inputs else pd.DataFrame()
    if not temporal_dist.empty:
        temporal_dist.columns = ["temporal_scope_type", "n"]
    year_min = int(pd.to_numeric(inputs.get("gold_start_year"), errors="coerce").min()) if not inputs.empty else int(runs["gold_start_year"].min())
    year_max = int(pd.to_numeric(inputs.get("gold_end_year"), errors="coerce").max()) if not inputs.empty else int(runs["gold_end_year"].max())

    findings = conservative_findings(summary_method, summary_type, errors)
    error_counts = (
        errors["error_type"].value_counts().rename_axis("error_type").reset_index(name="n")
        if not errors.empty
        else pd.DataFrame(columns=["error_type", "n"])
    )
    error_examples = errors[[
        "query_id", "rewrite_type", "query_type_normalized", "query_text", "method",
        "predicted_temporal_scope", "gold_start_year", "gold_end_year", "overlap_score", "error_type",
    ]].head(10)

    report = f"""# Bridge Rewrite Dataset v3 M0 Experiment Report

## 1. Objective

This M0-only experiment evaluates how the temporal scope inference framework performs across three implicit temporal query types: Type L (latent / no time clue), Type R (relative expression), and Type A (event-anchored).

The experiment asks:

```text
How well does the M0 temporal scope inference framework work across Latent, Relative, and Event-anchored queries?
```

## 2. Dataset

- Total rows: {total_rows}
- Source IDs: {source_count}
- Year range from gold labels: {year_min}-{year_max}

### Rewrite Type Distribution

{md_table(rewrite_dist)}

### Query Type Distribution

{md_table(qtype_dist)}

### Temporal Scope Type Distribution

{md_table(temporal_dist)}

## 3. Methods

Gold ranges are used only for evaluation, not for runtime candidate selection.

Candidate generation is adapter-based. For each query, the runner derives dataset-level year bounds and creates generic full, rolling, early/late, prefix/suffix, anchor-before/after, and event-before/after candidate scopes. It removes illegal and duplicate candidates.

The implemented methods are:

- `E2_stable_temporal_prior`: reuses the stable temporal prior scorer from `m0_final` when available.
- `FINAL_prior_guided_qualified_boundary`: reuses the final fused scorer from `m0_final` when available.

E6b/E7 are not separately reported in this experiment. FINAL is run with adapter-level neutral qualified-boundary feature values because Bridge v3 does not include real E7 qualified boundary features. Therefore FINAL results here should be read as diagnostic for adapter compatibility, not as a fully featured boundary-aware run.

## 4. Results

### Summary by Method

{md_table(summary_method)}

### Summary by Query Type

{md_table(summary_type)}

## 5. Findings

{chr(10).join(f'- {x}' for x in findings)}

These findings are conservative and specific to this adapter-based Bridge v3 M0 run.

## 6. Error Analysis

### Error Type Counts

{md_table(error_counts)}

### Example Error Cases

{md_table(error_examples)}

## 7. Implications for A7

This experiment directly supports A7's focus on implicit temporal scope inference across query types. It provides a type-specific M0 evaluation for latent, relative, and event-anchored rewrites, helping separate the difficulty of no-clue queries from relative-expression and event-anchored queries.

The results should be used as M0 evidence only. They do not evaluate final answer quality.

## 8. Limitations

- Candidate generation is adapter-based and does not use the original runtime M0 candidate generator.
- This experiment does not run full M1-M3 downstream retrieval, answer generation, RAGAS, or LLM judging.
- E6b/E7 are unavailable as full boundary-feature methods for this adapter. FINAL uses neutral adapter-level qualified-boundary features and is diagnostic.
- Results are specific to Bridge rewrite dataset v3.
- Gold labels are used for evaluation only, but dataset-level year bounds are derived from dataset metadata including gold/anchor/event years. This is a pragmatic adapter choice and should be noted in any paper.
"""

    report_path = md_dir / "bridge_v3_m0_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"wrote -> {csv_dir / 'bridge_v3_m0_summary_by_method.csv'}")
    print(f"wrote -> {csv_dir / 'bridge_v3_m0_summary_by_type.csv'}")
    print(f"wrote -> {csv_dir / 'bridge_v3_m0_error_cases.csv'}")
    print(f"wrote -> {report_path}")


if __name__ == "__main__":
    main()
