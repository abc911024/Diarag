from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import psycopg
from dotenv import load_dotenv


@dataclass(frozen=True)
class DbConfig:
    host: str
    port: int
    db: str
    user: str
    password: str

    @property
    def conninfo(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.db} "
            f"user={self.user} password={self.password}"
        )


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)


def config_from_args(args: argparse.Namespace) -> DbConfig:
    load_dotenv()
    return DbConfig(
        host=args.host or os.getenv("POSTGRES_HOST", "localhost"),
        port=args.port or int(os.getenv("POSTGRES_PORT", "5432")),
        db=args.db or os.getenv("POSTGRES_DB", "bridge_temporal"),
        user=args.user or os.getenv("POSTGRES_USER", "bridge"),
        password=args.password or os.getenv("POSTGRES_PASSWORD", "bridge"),
    )


def get_connection(args: argparse.Namespace) -> psycopg.Connection:
    return psycopg.connect(config_from_args(args).conninfo)


def quote_ident(identifier: str) -> str:
    if not identifier.replace("_", "").isalnum() or identifier[0].isdigit():
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return '"' + identifier.replace('"', '""') + '"'
