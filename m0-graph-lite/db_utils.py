from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import psycopg


def add_db_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--db", default="bridge_temporal")
    parser.add_argument("--user", default="bridge")
    parser.add_argument("--password", default="bridge")


def connect_db(args):
    return psycopg.connect(host=args.host, port=args.port, dbname=args.db, user=args.user, password=args.password)


def execute_sql_file(conn, path: Path) -> None:
    with conn.cursor() as cur:
        cur.execute(path.read_text(encoding="utf-8"))
    conn.commit()


def fetch_dataframe(conn, query: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    return pd.read_sql(query, conn, params=params)


def write_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} rows -> {path}")


def ensure_output_dirs() -> None:
    Path("m0-graph-lite/outputs/csv").mkdir(parents=True, exist_ok=True)
    Path("m0-graph-lite/outputs/markdown").mkdir(parents=True, exist_ok=True)
