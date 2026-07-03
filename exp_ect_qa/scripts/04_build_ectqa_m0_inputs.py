#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ANSWER_COLUMNS = ["answer", "answers", "reference_answer", "gold_answer"]
DATE_COLUMNS = ["supporting_evidence_dates", "evidence_dates", "date", "transcript_date", "fiscal_year", "year", "period"]


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        for col in df.columns:
            if col.lower() == name or name in col.lower():
                return col
    return None


def temporal_intent(query: str) -> str:
    q = (query or "").lower()
    if re.search(r"\b(19|20)\d{2}\b|q[1-4]|fy\s?\d{2,4}", q):
        return "explicit_time"
    if any(x in q for x in ["recent", "latest", "earlier", "later", "before", "after"]):
        return "relative_time"
    if any(x in q for x in ["trend", "change", "evolve", "over time", "growth", "decline"]):
        return "diachronic_trend"
    if any(x in q for x in ["compare", "compared", "versus", "vs"]):
        return "comparison"
    if any(x in q for x in ["after", "before", "following", "since"]):
        return "event_anchored"
    if any(x in q for x in ["current", "now", "latest"]):
        return "factual_time_sensitive"
    return "unknown"


def parse_years(value: Any) -> list[int]:
    if value is None or pd.isna(value):
        return []
    text = str(value)
    return sorted({int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus-csv", default="exp_ect_qa/processed/ectqa_analysis_focus.csv")
    parser.add_argument("--processed-dir", default="exp_ect_qa/processed")
    parser.add_argument("--dataset-name", default="austinmyc/ECT-QA")
    parser.add_argument("--output", default="exp_ect_qa/processed/ectqa_m0_inputs.csv")
    args = parser.parse_args()

    focus = pd.read_csv(args.focus_csv)
    combined = Path(args.processed_dir) / "raw_questions_train.csv"
    raw_paths = [combined] if combined.exists() else sorted(Path(args.processed_dir).glob("raw_*.csv"))
    raw_frames = [pd.read_csv(p) for p in raw_paths]
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    answer_col = first_col(raw, ANSWER_COLUMNS) if not raw.empty else None
    date_col = first_col(raw, DATE_COLUMNS) if not raw.empty else None

    rows = []
    for idx, row in focus.iterrows():
        raw_row = raw.iloc[idx] if idx < len(raw) else pd.Series(dtype=object)
        years = parse_years(raw_row.get(date_col)) if date_col else []
        gold_scope = [min(years), max(years)] if years else None
        answer = raw_row.get(answer_col) if answer_col else None
        has_focus = pd.notna(row.get("analysis_focus")) and float(row.get("analysis_focus_confidence") or 0) >= 0.35
        rows.append({
            "dataset_name": args.dataset_name,
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "temporal_intent": temporal_intent(row["query_text"]),
            "analysis_focus": row.get("analysis_focus"),
            "analysis_focus_source": row.get("analysis_focus_source"),
            "analysis_focus_confidence": row.get("analysis_focus_confidence"),
            "usable_for_m0_eval": bool(has_focus and gold_scope),
            "usable_for_downstream_eval": bool(pd.notna(answer) and str(answer).strip()),
            "reference_answer": answer,
            "gold_temporal_scope": json.dumps(gold_scope) if gold_scope else None,
            "acceptable_temporal_scopes": json.dumps([gold_scope]) if gold_scope else None,
            "gold_scope_source": date_col if gold_scope else None,
            "notes": "Gold scope derived from explicit temporal metadata." if gold_scope else "No reliable gold temporal scope derived.",
        })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
