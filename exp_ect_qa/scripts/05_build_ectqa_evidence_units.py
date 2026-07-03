#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


TEXT_COLUMNS = ["evidence", "context", "contexts", "transcript", "snippet", "passage", "text", "document"]
TIME_COLUMNS = ["transcript_date", "fiscal_year", "fiscal_quarter", "reporting_period", "timestamp", "date", "year", "period"]
DOC_COLUMNS = ["doc_id", "document_id", "source_doc_id", "transcript_id", "filing_id", "id"]
OUTPUT_COLUMNS = [
    "evidence_unit_id",
    "dataset_name",
    "query_id",
    "source_doc_id",
    "time_unit",
    "time_granularity",
    "evidence_text",
    "analysis_focus",
    "analysis_focus_source",
    "original_metadata_json",
    "usable_for_m0",
]


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        for col in df.columns:
            if col.lower() == name or name in col.lower():
                return col
    return None


def load_raw(processed_dir: Path) -> pd.DataFrame:
    frames = []
    preferred = []
    for name in ("raw_corpus_train.csv", "raw_questions_train.csv"):
        path = processed_dir / name
        if path.exists():
            preferred.append(path)
    paths = preferred or sorted(processed_dir.glob("raw_*.csv"))
    for path in paths:
        df = pd.read_csv(path)
        df["split"] = path.stem.removeprefix("raw_")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No raw_*.csv files found in {processed_dir}")
    return pd.concat(frames, ignore_index=True)


def parse_time_unit(value: Any) -> tuple[str | None, str]:
    if value is None or pd.isna(value):
        return None, "unknown"
    text = str(value)
    year = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    quarter = re.search(r"\b(Q[1-4])\b", text, re.I)
    if year and quarter:
        return f"{year.group(1)}-{quarter.group(1).upper()}", "quarter"
    if year:
        return year.group(1), "year"
    return text.strip() or None, "unknown"


def split_text(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value)
    # Preserve list-looking values as separate rough evidence units when possible.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
        except Exception:
            pass
    words = text.split()
    if len(words) <= 500:
        return [text]
    chunks = []
    step = 400
    overlap = 75
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + step]))
        start += step - overlap
    return chunks


def parse_evidence_list(value: Any) -> list[dict[str, Any]]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="exp_ect_qa/processed")
    parser.add_argument("--m0-inputs", default="exp_ect_qa/processed/ectqa_m0_inputs.csv")
    parser.add_argument("--dataset-name", default="austinmyc/ECT-QA")
    parser.add_argument("--output", default="exp_ect_qa/processed/ectqa_evidence_units.csv")
    args = parser.parse_args()

    raw = load_raw(Path(args.processed_dir))
    inputs = pd.read_csv(args.m0_inputs) if Path(args.m0_inputs).exists() else pd.DataFrame()
    text_col = first_col(raw, TEXT_COLUMNS)
    evidence_list_col = first_col(raw, ["evidence_list"])
    time_col = first_col(raw, TIME_COLUMNS)
    doc_col = first_col(raw, DOC_COLUMNS)
    if not text_col and not evidence_list_col:
        print("WARNING: no obvious evidence text column found; evidence unit file will be empty.")

    rows = []
    for idx, row in raw.iterrows():
        query_id = inputs.iloc[idx]["query_id"] if idx < len(inputs) else str(row.get("id") or f"{row.get('split')}_{idx}")
        focus = inputs.iloc[idx].get("analysis_focus") if idx < len(inputs) else None
        focus_source = inputs.iloc[idx].get("analysis_focus_source") if idx < len(inputs) else None
        evidence_records = parse_evidence_list(row.get(evidence_list_col)) if evidence_list_col else []
        if evidence_records:
            for unit_idx, ev in enumerate(evidence_records):
                time_unit, granularity = parse_time_unit(ev.get("year") or ev.get("date") or ev.get("period"))
                evidence_text = ev.get("evidence") or ev.get("text") or ev.get("content")
                source_doc_id = ev.get("ect_filename") or ev.get("doc_id") or ev.get("source_doc_id") or row.get(doc_col) if doc_col else ev.get("ect_filename")
                rows.append({
                    "evidence_unit_id": f"{query_id}__ev_{unit_idx:04d}",
                    "dataset_name": args.dataset_name,
                    "query_id": query_id,
                    "source_doc_id": source_doc_id,
                    "time_unit": time_unit,
                    "time_granularity": granularity,
                    "evidence_text": evidence_text,
                    "analysis_focus": focus,
                    "analysis_focus_source": focus_source,
                    "original_metadata_json": json.dumps({**row.dropna().to_dict(), "evidence_record": ev}, ensure_ascii=False, default=str),
                    "usable_for_m0": bool(time_unit and evidence_text),
                })
            continue

        time_unit, granularity = parse_time_unit(row.get(time_col)) if time_col else (None, "unknown")
        for unit_idx, text in enumerate(split_text(row.get(text_col)) if text_col else []):
            rows.append({
                "evidence_unit_id": f"{query_id}__ev_{unit_idx:04d}",
                "dataset_name": args.dataset_name,
                "query_id": query_id,
                "source_doc_id": row.get(doc_col) if doc_col else None,
                "time_unit": time_unit,
                "time_granularity": granularity,
                "evidence_text": text,
                "analysis_focus": focus,
                "analysis_focus_source": focus_source,
                "original_metadata_json": json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str),
                "usable_for_m0": bool(time_unit and text),
            })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
