#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from db_config import add_connection_args, config_from_args, quote_ident


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="qa")
    parser.add_argument("--output-dir", type=Path, default=Path("exports/qa"))
    add_connection_args(parser)
    args = parser.parse_args()
    schema = quote_ident(args.schema)
    cfg = config_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    views = ["v_bridge_dataset_overview", "v_m0_inputs_with_targets", "v_m0_metrics_summary"]
    import psycopg
    with psycopg.connect(cfg.conninfo) as conn:
        for view in views:
            df = pd.read_sql(f"SELECT * FROM {schema}.{quote_ident(view)}", conn)
            out = args.output_dir / f"{view}.csv"
            df.to_csv(out, index=False)
            print(f"exported {args.schema}.{view}: {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
