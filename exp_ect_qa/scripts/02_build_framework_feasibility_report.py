#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def decide_mode(summary: dict) -> tuple[str, list[str], list[str]]:
    available = []
    missing = []
    checks = {
        "query_text": summary.get("has_query_text"),
        "analysis_focus": summary.get("has_analysis_focus_source"),
        "time-indexed evidence units": summary.get("has_time_indexed_evidence") and summary.get("has_evidence_units"),
        "gold temporal scope": summary.get("can_derive_gold_temporal_scope"),
        "reference answer": summary.get("has_reference_answer"),
    }
    for name, ok in checks.items():
        (available if ok else missing).append(name)

    if all(checks[x] for x in ("query_text", "analysis_focus", "time-indexed evidence units", "gold temporal scope")):
        mode = "Mode A: Full M0 scope evaluation"
    elif all(checks[x] for x in ("query_text", "analysis_focus", "time-indexed evidence units")):
        mode = "Mode B: M0 scoring without quantitative scope accuracy"
    elif checks["query_text"] and checks["reference answer"] and checks["time-indexed evidence units"]:
        mode = "Mode C: Downstream answer validation"
    else:
        mode = "Mode D: Not usable"
    return mode, available, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-summary", default="exp_ect_qa/processed/schema_summary.json")
    parser.add_argument("--report-dir", default="exp_ect_qa/reports")
    args = parser.parse_args()

    summary_path = Path(args.schema_summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run 01_inspect_ectqa_schema.py first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mode, available, missing = decide_mode(summary)

    lines = [
        "# ECT-QA Framework Feasibility Report",
        "",
        f"Recommended mode: **{mode}**",
        "",
        "## Can ECT-QA Naturally Enter the Framework?",
        "",
        "ECT-QA can enter the framework only to the extent that its own fields or metadata support query text, analysis focus identification, time-indexed evidence units, and candidate temporal scopes. This report does not force Bridge-style company/ticker assumptions.",
        "",
        "## Analysis Focus",
        "",
        f"- Source type: `{summary.get('analysis_focus_source_type')}`",
        f"- Company/ticker optional fields present: `{summary.get('has_optional_company_or_ticker')}`",
        "- If explicit focus metadata is missing, the next step is rule-based query-derived focus extraction with confidence scoring.",
        "",
        "## Available Components",
        "",
        *[f"- {item}" for item in available],
        "",
        "## Missing or Uncertain Components",
        "",
        *[f"- {item}" for item in missing],
        "",
        "## Gold Temporal Scope",
        "",
        f"Can derive gold temporal scope without inventing it: **{summary.get('can_derive_gold_temporal_scope')}**.",
        "",
        "If this is false, do not report M0 acceptable accuracy. Use qualitative scope validation or downstream answer validation instead.",
        "",
        "## Recommended Next Steps",
        "",
        "1. Run `03_identify_analysis_focus.py` to create query-level analysis focus candidates.",
        "2. Build M0 inputs and evidence units only from fields actually present in ECT-QA.",
        "3. Generate candidate temporal scopes from available time units, not from gold labels.",
        "4. Run M0 scoring only for methods whose required features are available.",
        "5. If gold temporal scope is unavailable, report predicted scope distributions and downstream answer feasibility rather than M0 accuracy.",
        "",
        "## No Forced Conversion Assumptions",
        "",
        "Company/ticker fields are optional. They are not required and should not be invented. Gold temporal scopes should never be invented.",
    ]
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "framework_feasibility_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report -> {out}")


if __name__ == "__main__":
    main()
