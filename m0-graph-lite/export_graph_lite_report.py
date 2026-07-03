#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from db_utils import add_db_args, connect_db, fetch_dataframe


def md_table(df, max_rows: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).fillna("")
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c])[:120] for c in view.columns) + " |")
    return "\n".join(lines)


def generate_report(conn, output: Path) -> None:
    ticker = fetch_dataframe(conn, "SELECT * FROM m0_graph.v_ticker_year_timeline_summary ORDER BY total_chunks DESC")
    features = fetch_dataframe(conn, "SELECT * FROM m0_graph.v_candidate_graph_feature_summary ORDER BY window_type")
    by_type = fetch_dataframe(conn, "SELECT * FROM m0_graph.v_candidate_features_by_type ORDER BY rewrite_type, window_type")
    avg_cont = float(features["avg_year_continuity_score"].mean()) if not features.empty else 0.0
    avg_gap = float(features["avg_gap_count"].mean()) if not features.empty else 0.0
    avg_balance = float(features["avg_density_balance_score"].mean()) if not features.empty else 0.0
    interpretation = []
    if avg_cont > 0.9 and avg_gap < 0.5:
        interpretation.append("Content timeline is dense; simple continuity may not provide strong discrimination.")
    if not by_type.empty and by_type["avg_prefix_support_score"].std() > 0:
        interpretation.append("Position-based prefix/suffix features vary and may help directional scoring.")
    if avg_balance and avg_balance < 0.95:
        interpretation.append("Evidence distribution features vary and may distinguish stable windows from concentrated evidence windows.")
    if not interpretation:
        interpretation.append("Graph-lite v1 has limited variation; richer semantic relevance per year may be needed.")
    lines = [
        "# M0 Graph-Lite Report",
        "",
        "## Purpose",
        "Graph-lite is a PostgreSQL temporal structure feature layer for M0. It does not build a full graph database and does not use gold labels.",
        "",
        "## Data Sources",
        "- content.evidence_chunks",
        "- content.documents / content.document_entities where needed",
        "- runtime.m0_candidate_windows",
        "- runtime.m0_agent_runs",
        "- qa.m0_inputs / qa.bridge_records only for ticker/query metadata",
        "",
        "## Feature Definitions",
        "- year_continuity_score = years_with_evidence_count / window_length",
        "- gap_count = contiguous missing-evidence segments inside candidate window",
        "- max_gap_length = longest missing-evidence segment",
        "- normalized_start/end/center = candidate position within input timeline",
        "- prefix_support_score = 1 - normalized_start",
        "- suffix_support_score = normalized_end",
        "- coverage_density_score = average ticker-year density_score in window",
        "- density_balance_score = 1 / (1 + coefficient_of_variation(chunk_count))",
        "",
        "## Feature Summary",
        f"- avg continuity: {avg_cont:.4f}",
        f"- avg gap count: {avg_gap:.4f}",
        f"- avg density balance: {avg_balance:.4f}",
        "",
        "### Ticker Timeline Summary",
        md_table(ticker),
        "",
        "### Candidate Feature Summary",
        md_table(features),
        "",
        "### Candidate Features By Rewrite Type",
        md_table(by_type),
        "",
        "## Interpretation",
    ]
    lines.extend(f"- {item}" for item in interpretation)
    lines.extend(["", "## Recommended Next Step", "- Proceed to E4_graph_lite_scoring if features show variation; otherwise add semantic relevance per year before expanding graph work."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_db_args(parser)
    parser.add_argument("--output", type=Path, default=Path("m0-graph-lite/outputs/markdown/m0_graph_lite_report.md"))
    args = parser.parse_args()
    with connect_db(args) as conn:
        generate_report(conn, args.output)


if __name__ == "__main__":
    main()
