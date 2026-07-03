#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter

from psycopg.rows import dict_row

from db_config import add_connection_args, get_connection
from m0_agent_tools import get_content_available_years


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--m0-input-setting", default="query_with_corpus_metadata")
    parser.add_argument("--clear-existing", action="store_true")
    add_connection_args(parser)
    args = parser.parse_args()
    limit_sql = "LIMIT %s" if args.limit else ""
    params = [args.m0_input_setting]
    if args.limit:
        params.append(args.limit)

    with get_connection(args) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            if args.clear_existing:
                cur.execute("DELETE FROM runtime.qa_content_coverage WHERE m0_input_setting = %s", (args.m0_input_setting,))
            cur.execute(
                f"""
                SELECT
                    mi.m0_input_id, mi.bridge_id, mi.m0_input_setting,
                    mi.rewrite_type, br.rewritten_query, br.ticker,
                    ta.gold_start_year, ta.gold_end_year
                FROM qa.m0_inputs mi
                JOIN qa.bridge_records br ON br.bridge_id = mi.bridge_id
                LEFT JOIN qa.temporal_annotations ta ON ta.bridge_id = mi.bridge_id
                WHERE mi.m0_input_setting = %s
                ORDER BY mi.m0_input_id
                {limit_sql}
                """,
                tuple(params),
            )
            rows = cur.fetchall()
            statuses: Counter[str] = Counter()
            ratios = []
            for row in rows:
                ticker = row["ticker"]
                gold_start, gold_end = row["gold_start_year"], row["gold_end_year"]
                content = get_content_available_years(conn, ticker)
                content_years = set(content["content_available_years"])
                if gold_start is None or gold_end is None:
                    gold_years = []
                    covered = []
                    missing = []
                    ratio = 0.0
                    status = "missing_gold_target"
                else:
                    gold_years = list(range(int(gold_start), int(gold_end) + 1))
                    covered = sorted(set(gold_years) & content_years)
                    missing = [y for y in gold_years if y not in content_years]
                    ratio = len(covered) / len(gold_years) if gold_years else 0.0
                    if not content_years:
                        status = "no_content_for_ticker"
                    elif ratio == 1.0:
                        status = "full_gold_coverage"
                    elif ratio > 0:
                        status = "partial_gold_coverage"
                    else:
                        status = "no_gold_coverage"
                cur.execute(
                    """
                    INSERT INTO runtime.qa_content_coverage (
                        coverage_id, bridge_id, m0_input_id, ticker, rewritten_query, rewrite_type,
                        m0_input_setting, gold_start_year, gold_end_year, gold_years,
                        content_available_years, covered_gold_years, missing_gold_years,
                        matched_doc_count, matched_chunk_count, coverage_ratio, coverage_status,
                        can_run_m1_retrieval, can_run_temporal_retrieval, notes
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (coverage_id) DO UPDATE SET
                        content_available_years = EXCLUDED.content_available_years,
                        covered_gold_years = EXCLUDED.covered_gold_years,
                        missing_gold_years = EXCLUDED.missing_gold_years,
                        matched_doc_count = EXCLUDED.matched_doc_count,
                        matched_chunk_count = EXCLUDED.matched_chunk_count,
                        coverage_ratio = EXCLUDED.coverage_ratio,
                        coverage_status = EXCLUDED.coverage_status,
                        can_run_m1_retrieval = EXCLUDED.can_run_m1_retrieval,
                        can_run_temporal_retrieval = EXCLUDED.can_run_temporal_retrieval,
                        notes = EXCLUDED.notes
                    """,
                    (
                        f"{row['m0_input_id']}__coverage", row["bridge_id"], row["m0_input_id"],
                        ticker, row["rewritten_query"], row["rewrite_type"], row["m0_input_setting"],
                        gold_start, gold_end, json.dumps(gold_years), json.dumps(content["content_available_years"]),
                        json.dumps(covered), json.dumps(missing), content["matched_doc_count"],
                        content["matched_chunk_count"], ratio, status,
                        content["matched_chunk_count"] > 0, ratio > 0,
                        "Diagnostic table; gold used only for coverage/evaluation.",
                    ),
                )
                statuses[status] += 1
                ratios.append(ratio)
        conn.commit()
    total = sum(statuses.values())
    print(f"total evaluated: {total}")
    print(f"full coverage: {statuses['full_gold_coverage']}")
    print(f"partial coverage: {statuses['partial_gold_coverage']}")
    print(f"no coverage: {statuses['no_gold_coverage']}")
    print(f"no content for ticker: {statuses['no_content_for_ticker']}")
    print(f"average coverage ratio: {(sum(ratios) / len(ratios)) if ratios else 0:.4f}")


if __name__ == "__main__":
    main()
