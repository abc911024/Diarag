#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
import psycopg

from db_config import add_connection_args, config_from_args


def export_query(conn, sql: str, params: tuple, path: Path) -> pd.DataFrame:
    df = pd.read_sql(sql, conn, params=params)
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} rows -> {path}")
    return df


def main() -> None:
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="vector_probe_agent_v2")
    parser.add_argument("--m0-input-setting", default="query_with_corpus_metadata")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("data/runtime_exports/m0_debug"))
    add_connection_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = (args.method, args.m0_input_setting)
    with psycopg.connect(cfg.conninfo) as conn:
        export_query(
            conn,
            """
            SELECT *
            FROM runtime.v_selected_window_type_summary
            WHERE method = %s AND m0_input_setting = %s
            ORDER BY rewrite_type, count DESC
            """,
            params,
            args.output_dir / "selected_window_type_summary.csv",
        )
        export_query(
            conn,
            """
            SELECT
                candidate_generation_version,
                scoring_version,
                rewrite_type,
                m0_input_setting,
                COUNT(*) AS total_inputs,
                AVG(CASE WHEN has_exact_candidate THEN 1.0 ELSE 0.0 END) AS exact_candidate_rate,
                AVG(CASE WHEN has_acceptable_candidate THEN 1.0 ELSE 0.0 END) AS acceptable_candidate_rate,
                AVG(best_candidate_overlap) AS best_possible_avg_overlap,
                AVG(selected_overlap) AS selected_avg_overlap,
                AVG(best_candidate_overlap - COALESCE(selected_overlap, 0)) AS oracle_selection_gap
            FROM runtime.m0_candidate_oracle_analysis
            WHERE method = %s AND m0_input_setting = %s
            GROUP BY candidate_generation_version, scoring_version, rewrite_type, m0_input_setting
            ORDER BY rewrite_type
            """,
            params,
            args.output_dir / "candidate_oracle_summary.csv",
        )
        export_query(
            conn,
            """
            SELECT
                ar.agent_run_id,
                ar.query_text,
                ar.predicted_range,
                am.gold_range,
                cw.window_type AS selected_window_type,
                cw.relative_intent,
                ar.rationale,
                ar.tool_trace,
                am.overlap_score,
                am.mean_boundary_error,
                am.acceptable_accuracy
            FROM runtime.m0_agent_runs ar
            JOIN runtime.m0_agent_metrics am ON am.agent_run_id = ar.agent_run_id
            LEFT JOIN runtime.m0_candidate_windows cw ON cw.candidate_window_id = ar.selected_candidate_window_id
            WHERE ar.method = %s
              AND ar.m0_input_setting = %s
              AND ar.rewrite_type = 'B'
              AND COALESCE(am.acceptable_accuracy, false) = false
            ORDER BY am.mean_boundary_error DESC NULLS LAST
            LIMIT %s
            """,
            (args.method, args.m0_input_setting, args.limit),
            args.output_dir / "type_b_error_cases.csv",
        )
        export_query(
            conn,
            """
            SELECT
                ar.method,
                ar.rewrite_type,
                ar.m0_input_setting,
                AVG(ar.predicted_end_year - ar.predicted_start_year + 1) AS avg_pred_len,
                AVG(((am.gold_range ->> 1)::INTEGER - (am.gold_range ->> 0)::INTEGER + 1)) AS avg_gold_len,
                AVG((ar.predicted_end_year - ar.predicted_start_year + 1) - ((am.gold_range ->> 1)::INTEGER - (am.gold_range ->> 0)::INTEGER + 1)) AS avg_len_diff
            FROM runtime.m0_agent_runs ar
            JOIN runtime.m0_agent_metrics am ON am.agent_run_id = ar.agent_run_id
            WHERE ar.method = %s AND ar.m0_input_setting = %s
            GROUP BY ar.method, ar.rewrite_type, ar.m0_input_setting
            ORDER BY ar.rewrite_type
            """,
            params,
            args.output_dir / "predicted_vs_gold_length.csv",
        )
        export_query(
            conn,
            """
            SELECT
                ar.agent_run_id,
                ar.query_text,
                ar.predicted_range,
                am.gold_range,
                cw.window_type AS selected_window_type,
                ar.rationale,
                ar.tool_trace,
                am.overlap_score,
                am.mean_boundary_error,
                am.acceptable_accuracy
            FROM runtime.m0_agent_runs ar
            JOIN runtime.m0_agent_metrics am ON am.agent_run_id = ar.agent_run_id
            LEFT JOIN runtime.m0_candidate_windows cw ON cw.candidate_window_id = ar.selected_candidate_window_id
            WHERE ar.method = %s AND ar.m0_input_setting = %s
            ORDER BY am.mean_boundary_error DESC NULLS LAST
            LIMIT 20
            """,
            params,
            args.output_dir / "worst_m0_errors.csv",
        )


if __name__ == "__main__":
    main()
