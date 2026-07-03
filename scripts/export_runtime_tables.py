#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import psycopg

from db_config import add_connection_args, config_from_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/runtime_exports"))
    add_connection_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "qa_content_coverage": "runtime.qa_content_coverage",
        "qa_content_coverage_summary": "runtime.v_qa_content_coverage_summary",
        "m0_agent_results": "runtime.v_m0_agent_results",
        "m0_agent_summary": "runtime.v_m0_agent_summary",
        "probe_summary": "runtime.v_probe_summary",
        "candidate_oracle_analysis": "runtime.m0_candidate_oracle_analysis",
        "selected_window_type_summary": "runtime.v_selected_window_type_summary",
        "candidate_oracle_summary": "runtime.v_candidate_oracle_summary",
        "m0_length_bias": "runtime.v_m0_length_bias",
        "type_b_relative_error_summary": "runtime.v_type_b_relative_error_summary",
    }
    with psycopg.connect(cfg.conninfo) as conn:
        for name, relation in exports.items():
            df = pd.read_sql(f"SELECT * FROM {relation}", conn)
            out = args.output_dir / f"{name}.csv"
            df.to_csv(out, index=False)
            print(f"exported {relation}: {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
