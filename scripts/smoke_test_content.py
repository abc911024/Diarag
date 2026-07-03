#!/usr/bin/env python3
from __future__ import annotations

import argparse

from db_config import add_connection_args, get_connection, quote_ident


def scalar(cur, sql: str, params=None):
    cur.execute(sql, params or ())
    return cur.fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="content")
    add_connection_args(parser)
    args = parser.parse_args()
    schema = quote_ident(args.schema)
    with get_connection(args) as conn:
        with conn.cursor() as cur:
            exists = scalar(cur, "SELECT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = %s)", (args.schema,))
            assert exists, f"schema {args.schema} does not exist"
            for table in ("raw_documents", "documents", "document_entities", "evidence_chunks"):
                count = scalar(cur, f"SELECT COUNT(*) FROM {schema}.{quote_ident(table)}")
                print(f"{args.schema}.{table}: {count}")
            cur.execute(f"SELECT * FROM {schema}.v_content_document_overview LIMIT 1")
            cur.fetchone()
            cur.execute(f"SELECT * FROM {schema}.v_content_entity_year_coverage LIMIT 1")
            cur.fetchone()
            cur.execute(f"SELECT * FROM {schema}.v_content_chunk_overview LIMIT 1")
            cur.fetchone()
            cur.execute(f"""
                SELECT chunk_id, chunk_text
                FROM {schema}.evidence_chunks
                ORDER BY embedding <=> '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'::vector
                LIMIT 5
            """)
            cur.fetchall()
            bad_dim = scalar(cur, f"SELECT COUNT(*) FROM {schema}.evidence_chunks WHERE vector_dims(embedding) <> 8")
            assert bad_dim == 0, "some sampled embeddings are not dimension 8"
            missing_docs = scalar(cur, f"""
                SELECT COUNT(*)
                FROM {schema}.evidence_chunks ec
                LEFT JOIN {schema}.documents d ON d.doc_id = ec.doc_id
                WHERE d.doc_id IS NULL
            """)
            assert missing_docs == 0, "some chunks reference missing documents"
    print("Content smoke test passed")


if __name__ == "__main__":
    main()
