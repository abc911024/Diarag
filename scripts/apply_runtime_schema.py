#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from db_config import add_connection_args, get_connection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-debug", action="store_true")
    add_connection_args(parser)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sql_paths = [
        root / "postgres" / "init" / "006_schema_runtime.sql",
        root / "postgres" / "init" / "007_views_runtime.sql",
    ]
    if args.include_debug:
        sql_paths.append(root / "postgres" / "init" / "008_views_m0_debug.sql")
    with get_connection(args) as conn:
        with conn.cursor() as cur:
            for path in sql_paths:
                cur.execute(path.read_text(encoding="utf-8"))
                print(f"applied {path}")
        conn.commit()
    print("Runtime schema applied successfully")


if __name__ == "__main__":
    main()
