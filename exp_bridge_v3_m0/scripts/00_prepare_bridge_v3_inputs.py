#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


TYPE_MAP = {
    "type l": "latent",
    "l": "latent",
    "latent": "latent",
    "type r": "relative",
    "r": "relative",
    "relative": "relative",
    "type a": "event_anchored",
    "a": "event_anchored",
    "event-anchored": "event_anchored",
    "event anchored": "event_anchored",
    "event_anchored": "event_anchored",
}


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def parse_literal(value: Any) -> tuple[Any, str]:
    if value is None or pd.isna(value):
        return None, ""
    if isinstance(value, (list, tuple)):
        return value, ""
    text = str(value).strip()
    if not text:
        return None, ""
    try:
        return ast.literal_eval(text), ""
    except Exception as exc:
        return value, f"literal_parse_failed: {exc}"


def normalize_type(value: Any) -> str:
    key = str(value or "").strip().lower()
    return TYPE_MAP.get(key, key.replace(" ", "_") if key else "unknown")


def as_year(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except Exception:
        text = str(value)
        for token in text.replace("/", "-").split("-"):
            if token.isdigit() and len(token) == 4:
                return int(token)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="m0_final/bridge_rewrite_dataset_v3.csv")
    parser.add_argument("--output", default="exp_bridge_v3_m0/data/bridge_v3_m0_inputs.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    warnings: list[str] = []

    df["query_id"] = df.get("bridge_id")
    df["query_text"] = df.get("rewritten_query")
    df["target_subject"] = df.get("ticker")
    df["target_name"] = df.get("stock_name")
    df["gold_temporal_scope"] = df.get("gold_range")
    df["acceptable_temporal_scopes"] = df.get("acceptable_ranges")
    df["gold_start_time"] = df.get("gold_start_date")
    df["gold_end_time"] = df.get("gold_end_date")
    df["query_type_normalized"] = df.get("rewrite_type").apply(normalize_type)

    gold_ranges = []
    acceptable_ranges = []
    gold_starts = []
    gold_ends = []
    parse_warnings = []

    for idx, row in df.iterrows():
        gold, gold_warning = parse_literal(row.get("gold_range"))
        acceptable, acc_warning = parse_literal(row.get("acceptable_ranges"))
        gold_ranges.append(gold)
        acceptable_ranges.append(acceptable)
        start = end = None
        if isinstance(gold, (list, tuple)) and len(gold) >= 2:
            start, end = as_year(gold[0]), as_year(gold[1])
        if start is None:
            start = as_year(row.get("gold_start_date"))
        if end is None:
            end = as_year(row.get("gold_end_date"))
        gold_starts.append(start)
        gold_ends.append(end)
        row_warnings = [w for w in [gold_warning, acc_warning] if w]
        parse_warnings.append("; ".join(row_warnings))
        if row_warnings:
            warnings.append(f"row={idx} bridge_id={row.get('bridge_id')}: {'; '.join(row_warnings)}")

    df["gold_range_parsed"] = [json.dumps(x) if isinstance(x, (list, tuple)) else x for x in gold_ranges]
    df["acceptable_ranges_parsed"] = [json.dumps(x) if isinstance(x, (list, tuple)) else x for x in acceptable_ranges]
    df["gold_start_year"] = gold_starts
    df["gold_end_year"] = gold_ends
    df["parse_warning"] = parse_warnings

    out = ensure_parent(args.output)
    df.to_csv(out, index=False)

    copy_path = out.parent / "bridge_rewrite_dataset_v3.csv"
    if input_path.resolve() != copy_path.resolve():
        df_original = pd.read_csv(input_path)
        df_original.to_csv(copy_path, index=False)

    print(f"loaded rows: {len(df)}")
    print(f"rewrite_type distribution: {df['rewrite_type'].value_counts(dropna=False).to_dict()}")
    print(f"query_type_normalized distribution: {df['query_type_normalized'].value_counts(dropna=False).to_dict()}")
    print(f"parse warnings: {len(warnings)}")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
