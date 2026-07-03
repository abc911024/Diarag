#!/usr/bin/env python3
from __future__ import annotations

import argparse

from db_config import add_connection_args, get_connection, quote_ident

FORBIDDEN_KEYS = [
    "gold_range",
    "acceptable_ranges",
    "gold_start_date",
    "gold_end_date",
    "source_temporal_label",
    "temporal_scope_type",
]


def scalar(cur, sql: str, params=None):
    cur.execute(sql, params or ())
    return cur.fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="qa")
    add_connection_args(parser)
    args = parser.parse_args()
    schema = quote_ident(args.schema)
    with get_connection(args) as conn:
        with conn.cursor() as cur:
            exists = scalar(cur, "SELECT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = %s)", (args.schema,))
            assert exists, f"schema {args.schema} does not exist"
            for table in ("raw_qa_items", "bridge_records", "m0_inputs", "temporal_annotations"):
                count = scalar(cur, f"SELECT COUNT(*) FROM {schema}.{quote_ident(table)}")
                print(f"{args.schema}.{table}: {count}")
            cur.execute(f"SELECT * FROM {schema}.v_bridge_dataset_overview LIMIT 1")
            cur.fetchone()
            cur.execute(f"SELECT * FROM {schema}.v_m0_inputs_with_targets LIMIT 1")
            cur.fetchone()
            missing_targets = scalar(cur, f"""
                SELECT COUNT(*)
                FROM {schema}.m0_inputs mi
                LEFT JOIN {schema}.temporal_annotations ta ON ta.bridge_id = mi.bridge_id
                WHERE ta.bridge_id IS NULL
            """)
            assert missing_targets == 0, "some m0_inputs do not have temporal annotation targets"
            for key in FORBIDDEN_KEYS:
                count = scalar(cur, f"SELECT COUNT(*) FROM {schema}.m0_inputs WHERE m0_input_json ? %s", (key,))
                assert count == 0, f"m0_input_json contains forbidden key: {key}"
    print("QA smoke test passed")


if __name__ == "__main__":
    main()
