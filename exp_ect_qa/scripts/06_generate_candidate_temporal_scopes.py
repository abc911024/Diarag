#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "candidate_scope_id",
    "query_id",
    "analysis_focus",
    "start_time_unit",
    "end_time_unit",
    "scope_type",
    "time_granularity",
    "scope_length",
    "scope_generation_source",
]


def year_value(value) -> int | None:
    try:
        text = str(value)
        return int(text[:4])
    except Exception:
        return None


def add_scope(rows, query_id, focus, start, end, scope_type, granularity, source):
    rows.append({
        "candidate_scope_id": f"{query_id}__scope_{len(rows):04d}",
        "query_id": query_id,
        "analysis_focus": focus,
        "start_time_unit": start,
        "end_time_unit": end,
        "scope_type": scope_type,
        "time_granularity": granularity,
        "scope_length": end - start + 1,
        "scope_generation_source": source,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-units", default="exp_ect_qa/processed/ectqa_evidence_units.csv")
    parser.add_argument("--output", default="exp_ect_qa/processed/ectqa_candidate_temporal_scopes.csv")
    args = parser.parse_args()

    ev = pd.read_csv(args.evidence_units)
    if ev.empty:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(out, index=False)
        print(f"WARNING: evidence unit file is empty; wrote empty scopes -> {out}")
        return
    ev = ev[ev["usable_for_m0"].astype(str).str.lower().isin(["true", "1"])]
    rows = []
    for query_id, group in ev.groupby("query_id"):
        years = sorted({year_value(x) for x in group["time_unit"].dropna()})
        years = [y for y in years if y is not None]
        if not years:
            continue
        focus = group["analysis_focus"].dropna().iloc[0] if group["analysis_focus"].notna().any() else None
        granularity = "year"
        add_scope(rows, query_id, focus, min(years), max(years), "full_available_scope", granularity, "evidence_availability")
        for y in years:
            add_scope(rows, query_id, focus, y, y, "single_time_unit", granularity, "evidence_availability")
        for width in (3, 5, 8):
            if len(years) >= width:
                for start in range(min(years), max(years) - width + 2):
                    end = start + width - 1
                    if start in years or end in years:
                        add_scope(rows, query_id, focus, start, end, f"rolling_{width}", granularity, "evidence_availability")
        if len(years) >= 3:
            add_scope(rows, query_id, focus, max(years) - 2, max(years), "recent_3", granularity, "evidence_availability")
        if len(years) >= 5:
            add_scope(rows, query_id, focus, max(years) - 4, max(years), "recent_5", granularity, "evidence_availability")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).drop_duplicates(subset=["query_id", "start_time_unit", "end_time_unit", "scope_type"]).to_csv(out, index=False)
    print(f"wrote {len(rows)} candidate scopes -> {out}")


if __name__ == "__main__":
    main()
