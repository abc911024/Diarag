#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


FIELD_CANDIDATES = {
    "query_text": ["question", "query", "query_text", "qa_question", "prompt"],
    "answer": ["answer", "answers", "reference_answer", "gold_answer"],
    "evidence_text": ["evidence", "context", "contexts", "transcript", "snippet", "passage", "text", "document", "raw_content", "cleaned_content", "content"],
    "supporting_evidence": ["supporting_evidence", "support", "evidence_ids", "supporting_facts"],
    "date": ["date", "timestamp", "transcript_date", "filing_date", "period", "reporting_period"],
    "fiscal_year": ["fiscal_year", "year", "fy"],
    "fiscal_quarter": ["fiscal_quarter", "quarter", "fq"],
    "company": ["company", "company_name", "ticker", "symbol"],
    "topic": ["topic", "event", "metric", "title", "doc_title", "document_title"],
    "doc_id": ["doc_id", "document_id", "source_doc_id", "transcript_id", "filing_id"],
}


def read_raw_files(processed_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    preferred = []
    for name in ("raw_questions_train.csv", "raw_corpus_train.csv"):
        path = processed_dir / name
        if path.exists():
            preferred.append(path)
    paths = preferred or sorted(processed_dir.glob("raw_*.csv"))
    for path in paths:
        split = path.stem.removeprefix("raw_")
        frames[split] = pd.read_csv(path)
    return frames


def matching_columns(columns: list[str], candidates: list[str]) -> list[str]:
    lowered = {c.lower(): c for c in columns}
    matches = []
    for candidate in candidates:
        for lower, original in lowered.items():
            if lower == candidate or candidate in lower:
                matches.append(original)
    return sorted(set(matches))


def sample_has_temporal_values(df: pd.DataFrame, columns: list[str]) -> bool:
    if not columns:
        return False
    pattern = re.compile(r"\b(19|20)\d{2}\b|Q[1-4]|FY\s?\d{2,4}", re.I)
    for col in columns:
        sample = " ".join(str(x) for x in df[col].dropna().head(50).tolist())
        if pattern.search(sample):
            return True
    return False


def infer_summary(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    all_columns = sorted({col for df in frames.values() for col in df.columns})
    first_df = next(iter(frames.values())) if frames else pd.DataFrame()
    found = {key: matching_columns(all_columns, vals) for key, vals in FIELD_CANDIDATES.items()}

    has_query = bool(found["query_text"])
    focus_sources = found["company"] + found["topic"]
    has_focus_source = bool(focus_sources) or has_query
    if focus_sources:
        focus_type = "metadata"
    elif has_query:
        focus_type = "query_derived"
    else:
        focus_type = "unavailable"

    temporal_cols = found["date"] + found["fiscal_year"] + found["fiscal_quarter"]
    has_time = bool(temporal_cols) or sample_has_temporal_values(first_df, found["evidence_text"])
    has_evidence = bool(found["evidence_text"] or found["supporting_evidence"])
    has_reference_answer = bool(found["answer"])
    has_support_dates = bool(temporal_cols and found["supporting_evidence"])
    can_gold = has_support_dates
    usable_m0 = bool(has_query and has_focus_source and has_time and has_evidence and can_gold)
    usable_downstream = bool(has_query and has_reference_answer and has_evidence)
    if usable_m0:
        mode = "m0_scope_eval"
    elif usable_downstream:
        mode = "downstream_answer_eval"
    elif has_query and has_focus_source and has_time and has_evidence:
        mode = "qualitative_only"
    else:
        mode = "qualitative_only"

    return {
        "has_query_text": has_query,
        "has_analysis_focus_source": has_focus_source,
        "analysis_focus_source_type": focus_type if focus_type != "query_derived" else "query_derived",
        "has_time_indexed_evidence": has_time,
        "has_evidence_units": has_evidence,
        "has_reference_answer": has_reference_answer,
        "has_supporting_evidence_dates": has_support_dates,
        "has_optional_company_or_ticker": bool(found["company"]),
        "can_derive_gold_temporal_scope": can_gold,
        "usable_for_m0_scope_eval": usable_m0,
        "usable_for_downstream_answer_eval": usable_downstream,
        "recommended_evaluation_mode": mode,
        "column_matches": found,
        "all_columns": all_columns,
        "inspection_notes": "Column-based inspection only; uncertain mappings should be verified against examples.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="exp_ect_qa/processed")
    parser.add_argument("--report-dir", default="exp_ect_qa/reports")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    frames = read_raw_files(processed_dir)
    if not frames:
        raise FileNotFoundError(f"No raw_*.csv files found in {processed_dir}. Run 00_import_ectqa.py first.")

    summary = infer_summary(frames)
    (processed_dir / "schema_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# ECT-QA Schema Inspection Report",
        "",
        "## Split Overview",
        "",
    ]
    for split, df in frames.items():
        lines += [
            f"### {split}",
            "",
            f"- Rows: {len(df)}",
            f"- Columns: {list(df.columns)}",
            "",
            "First two rows:",
            "",
            "```json",
            df.head(2).to_json(orient="records", force_ascii=False, indent=2),
            "```",
            "",
        ]

    lines += [
        "## Framework Component Inspection",
        "",
        f"1. Query text available: **{summary['has_query_text']}** via `{summary['column_matches']['query_text']}`.",
        f"2. Analysis focus source: **{summary['analysis_focus_source_type']}** via metadata candidates `{summary['column_matches']['company'] + summary['column_matches']['topic']}`; query-derived fallback is possible: `{summary['has_query_text']}`.",
        f"3. Time-indexed evidence available: **{summary['has_time_indexed_evidence']}** via `{summary['column_matches']['date'] + summary['column_matches']['fiscal_year'] + summary['column_matches']['fiscal_quarter']}` or temporal values detected in evidence text.",
        f"4. Evidence units available: **{summary['has_evidence_units']}** via `{summary['column_matches']['evidence_text'] + summary['column_matches']['supporting_evidence']}`.",
        f"5. Reference answers available: **{summary['has_reference_answer']}** via `{summary['column_matches']['answer']}`.",
        f"6. Supporting evidence dates/doc ids available: dates={summary['has_supporting_evidence_dates']}, doc candidates=`{summary['column_matches']['doc_id']}`.",
        f"7. Gold temporal scope derivable without invention: **{summary['can_derive_gold_temporal_scope']}**.",
        f"8. Usability: M0 scope eval={summary['usable_for_m0_scope_eval']}; downstream answer eval={summary['usable_for_downstream_answer_eval']}; recommended mode=`{summary['recommended_evaluation_mode']}`.",
        "",
        "## Caution",
        "",
        "This report does not require company or ticker fields. If no explicit metadata source exists, analysis focus may be query-derived, but low-confidence examples should be filtered or marked accordingly.",
    ]
    (report_dir / "schema_inspection_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote schema summary -> {processed_dir / 'schema_summary.json'}")
    print(f"wrote report -> {report_dir / 'schema_inspection_report.md'}")


if __name__ == "__main__":
    main()
