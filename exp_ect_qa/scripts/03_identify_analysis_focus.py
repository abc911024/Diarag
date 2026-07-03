#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


QUESTION_COLUMNS = ["question", "query", "query_text", "qa_question", "prompt"]
METADATA_FOCUS_COLUMNS = ["company", "company_name", "ticker", "symbol", "topic", "event", "metric", "product", "policy", "title", "doc_title", "document_title"]
TEMPORAL_PATTERNS = [
    r"\bover time\b", r"\bin \d{4}\b", r"\bduring .*?\b", r"\brecently\b",
    r"\bbetween \d{4} and \d{4}\b", r"\bfrom \d{4} to \d{4}\b",
    r"\bQ[1-4]\b", r"\bFY\s?\d{2,4}\b",
]
GENERIC_WORDS = {
    "what", "how", "why", "when", "where", "which", "did", "does", "do", "was",
    "were", "is", "are", "the", "a", "an", "of", "to", "for", "in", "on",
    "about", "describe", "explain", "according", "based", "call", "earnings",
}


def first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        for lower, original in lowered.items():
            if lower == candidate or candidate in lower:
                return original
    return None


def query_focus(question: str) -> tuple[str | None, float, str]:
    text = question or ""
    for pattern in TEMPORAL_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9&%$.\- ]+", " ", text)
    words = [w for w in text.split() if w.lower() not in GENERIC_WORDS and len(w) > 2]
    focus = " ".join(words[:8]).strip()
    if not focus:
        return None, 0.0, "No reliable query-derived focus phrase."
    confidence = 0.55 if len(words) >= 2 else 0.35
    return focus, confidence, "Rule-based query-derived focus."


def metadata_focus(row: pd.Series) -> tuple[str | None, str | None]:
    parts = []
    sources = []
    for col in METADATA_FOCUS_COLUMNS:
        matches = [c for c in row.index if c.lower() == col or col in c.lower()]
        for match in matches:
            value = row.get(match)
            if pd.notna(value) and str(value).strip():
                parts.append(str(value).strip())
                sources.append(match)
    if not parts:
        return None, None
    deduped = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return " ".join(deduped[:3]), ",".join(sources[:3])


def load_all(processed_dir: Path) -> pd.DataFrame:
    frames = []
    combined = processed_dir / "raw_questions_train.csv"
    question_paths = [combined] if combined.exists() else sorted(processed_dir.glob("raw_questions*.csv"))
    paths = question_paths or sorted(processed_dir.glob("raw_*.csv"))
    for path in paths:
        df = pd.read_csv(path)
        df["split"] = path.stem.removeprefix("raw_")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No raw_*.csv files found in {processed_dir}.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="exp_ect_qa/processed")
    parser.add_argument("--output", default="exp_ect_qa/processed/ectqa_analysis_focus.csv")
    args = parser.parse_args()

    df = load_all(Path(args.processed_dir))
    q_col = first_col(df, QUESTION_COLUMNS)
    if not q_col:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "query_id",
            "query_text",
            "analysis_focus",
            "analysis_focus_source",
            "analysis_focus_confidence",
            "focus_extraction_notes",
            "original_metadata_json",
        ]).to_csv(out, index=False)
        report = out.parent.parent / "reports" / "analysis_focus_report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# Analysis Focus Identification Report\n\n"
            "No question/query column was found in the currently imported files. "
            "This usually means only the ECT-QA `corpus` config was imported. "
            "Rerun `00_import_ectqa.py`; the importer now includes a fallback for the `questions` JSON files.\n",
            encoding="utf-8",
        )
        print(f"WARNING: no question/query column found. Wrote empty output -> {out}")
        return
    df = df[df[q_col].notna() & df[q_col].astype(str).str.strip().ne("")]

    rows = []
    for idx, row in df.iterrows():
        question = str(row.get(q_col) or "")
        meta, meta_source = metadata_focus(row)
        q_focus, q_conf, q_note = query_focus(question)
        if meta and q_focus:
            focus = f"{meta} {q_focus}"
            source = f"metadata+query_derived:{meta_source}"
            confidence = min(0.90, max(0.70, q_conf + 0.20))
            notes = f"Combined metadata focus with query-derived phrase. {q_note}"
        elif meta:
            focus = meta
            source = f"metadata:{meta_source}"
            confidence = 0.75
            notes = "Metadata focus used."
        elif q_focus:
            focus = q_focus
            source = "query_derived"
            confidence = q_conf
            notes = q_note
        else:
            focus = None
            source = "unavailable"
            confidence = 0.0
            notes = "No reliable focus source found."
        query_id = str(row.get("id") or row.get("qid") or row.get("question_id") or f"{row.get('split')}_{idx}")
        rows.append({
            "query_id": query_id,
            "query_text": question,
            "analysis_focus": focus,
            "analysis_focus_source": source,
            "analysis_focus_confidence": confidence,
            "focus_extraction_notes": notes,
            "original_metadata_json": json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str),
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
