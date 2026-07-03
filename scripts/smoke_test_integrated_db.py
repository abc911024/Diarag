#!/usr/bin/env python3
from __future__ import annotations

import argparse

from db_config import add_connection_args, get_connection

QA_TABLES = [
    "raw_qa_items", "bridge_candidates", "bridge_records", "temporal_annotations",
    "entity_coverage", "m0_inputs", "m0_predictions", "m0_metrics", "qa_pipeline_runs",
]
CONTENT_TABLES = [
    "raw_documents", "documents", "document_entities", "evidence_chunks",
    "content_build_runs", "content_retrieval_runs",
]


def scalar(cur, sql: str, params=None):
    cur.execute(sql, params or ())
    return cur.fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_connection_args(parser)
    args = parser.parse_args()
    with get_connection(args) as conn:
        with conn.cursor() as cur:
            for schema in ("qa", "content"):
                exists = scalar(cur, "SELECT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = %s)", (schema,))
                assert exists, f"schema {schema} does not exist"
            vector_exists = scalar(cur, "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            assert vector_exists, "pgvector extension is not installed"
            for table in QA_TABLES:
                exists = scalar(cur, """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'qa' AND table_name = %s
                    )
                """, (table,))
                assert exists, f"missing qa table {table}"
            for table in CONTENT_TABLES:
                exists = scalar(cur, """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'content' AND table_name = %s
                    )
                """, (table,))
                assert exists, f"missing content table {table}"
            qa_bridge_records = scalar(cur, "SELECT COUNT(*) FROM qa.bridge_records")
            qa_m0_inputs = scalar(cur, "SELECT COUNT(*) FROM qa.m0_inputs")
            content_documents = scalar(cur, "SELECT COUNT(*) FROM content.documents")
            content_chunks = scalar(cur, "SELECT COUNT(*) FROM content.evidence_chunks")
            cross_views = scalar(cur, """
                SELECT COUNT(*)
                FROM information_schema.views
                WHERE table_schema IN ('qa', 'content')
                  AND (
                    table_name ILIKE %s
                    OR table_name ILIKE %s
                    OR table_name ILIKE %s
                    OR (view_definition ILIKE %s AND view_definition ILIKE %s)
                  )
            """, ("%qa_content%", "%content_qa%", "%coverage_match%", "%qa.%", "%content.%"))
            assert cross_views == 0, "unexpected QA-content joined view exists"
            print(f"qa.bridge_records: {qa_bridge_records}")
            print(f"qa.m0_inputs: {qa_m0_inputs}")
            print(f"content.documents: {content_documents}")
            print(f"content.evidence_chunks: {content_chunks}")
    print("Integrated database smoke test passed")


if __name__ == "__main__":
    main()
