#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from db_config import add_connection_args, get_connection, quote_ident


def read_jsonl(path: Path, optional: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"Warning: {'optional ' if optional else ''}file {path} missing; skipping")
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        print(f"Warning: optional file {path} missing or empty; skipping")
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def vector_literal(values: list[float]) -> str:
    if len(values) != 8:
        raise ValueError(f"Expected embedding dimension 8, got {len(values)}")
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def parse_jsonish(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (list, dict)):
        return value
    return json.loads(value)


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/content"))
    parser.add_argument("--schema", default="content")
    add_connection_args(parser)
    args = parser.parse_args()
    schema = quote_ident(args.schema)

    raw_documents = read_jsonl(args.data_dir / "raw_documents.jsonl")
    documents = read_jsonl(args.data_dir / "documents.jsonl")
    entities = read_jsonl(args.data_dir / "document_entities.jsonl")
    chunks = read_jsonl(args.data_dir / "evidence_chunks.jsonl")
    runs = read_jsonl(args.data_dir / "content_build_runs.jsonl", optional=True)
    retrieval_rows = read_csv(args.data_dir / "content_retrieval_demo_results.csv")
    counts = {name: 0 for name in [
        "raw_documents", "documents", "document_entities", "evidence_chunks",
        "content_build_runs", "content_retrieval_runs",
    ]}

    with get_connection(args) as conn:
        with conn.cursor() as cur:
            for r in raw_documents:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.raw_documents (doc_id, source_type, raw_text, raw_json, text_length)
                    VALUES (%s,%s,%s,%s::jsonb,%s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        source_type = EXCLUDED.source_type,
                        raw_text = EXCLUDED.raw_text,
                        raw_json = EXCLUDED.raw_json,
                        text_length = EXCLUDED.text_length
                    """,
                    (r["doc_id"], r.get("source_type"), r["raw_text"], dumps(r.get("raw_json")), r.get("text_length")),
                )
                counts["raw_documents"] += 1
            for r in documents:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.documents (
                        doc_id, title, source_name, source_type, document_year, document_year_source,
                        mentioned_years, candidate_years, title_extraction_status, year_extraction_status,
                        metadata_quality_status, metadata
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        source_name = EXCLUDED.source_name,
                        source_type = EXCLUDED.source_type,
                        document_year = EXCLUDED.document_year,
                        document_year_source = EXCLUDED.document_year_source,
                        mentioned_years = EXCLUDED.mentioned_years,
                        candidate_years = EXCLUDED.candidate_years,
                        title_extraction_status = EXCLUDED.title_extraction_status,
                        year_extraction_status = EXCLUDED.year_extraction_status,
                        metadata_quality_status = EXCLUDED.metadata_quality_status,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        r["doc_id"], r.get("title"), r.get("source_name"), r.get("source_type"),
                        r.get("document_year"), r.get("document_year_source"), dumps(r.get("mentioned_years", [])),
                        dumps(r.get("candidate_years", [])), r.get("title_extraction_status"),
                        r.get("year_extraction_status"), r.get("metadata_quality_status"), dumps(r.get("metadata", {})),
                    ),
                )
                counts["documents"] += 1
            for r in entities:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.document_entities (
                        doc_entity_id, doc_id, entity_text, ticker, entity_type, match_source,
                        match_confidence, context_snippet
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (doc_entity_id) DO UPDATE SET
                        entity_text = EXCLUDED.entity_text,
                        ticker = EXCLUDED.ticker,
                        entity_type = EXCLUDED.entity_type,
                        match_source = EXCLUDED.match_source,
                        match_confidence = EXCLUDED.match_confidence,
                        context_snippet = EXCLUDED.context_snippet
                    """,
                    (
                        r["doc_entity_id"], r["doc_id"], r.get("entity_text"), r.get("ticker"),
                        r.get("entity_type"), r.get("match_source"), r.get("match_confidence"),
                        r.get("context_snippet"),
                    ),
                )
                counts["document_entities"] += 1
            for r in chunks:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.evidence_chunks (
                        chunk_id, doc_id, chunk_index, chunk_text, chunk_char_start, chunk_char_end,
                        chunk_token_estimate, document_year, document_year_source, mentioned_years,
                        mentioned_tickers, source_name, source_type, title, chunk_quality_status,
                        metadata, embedding
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s::jsonb,%s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        chunk_char_start = EXCLUDED.chunk_char_start,
                        chunk_char_end = EXCLUDED.chunk_char_end,
                        chunk_token_estimate = EXCLUDED.chunk_token_estimate,
                        document_year = EXCLUDED.document_year,
                        document_year_source = EXCLUDED.document_year_source,
                        mentioned_years = EXCLUDED.mentioned_years,
                        mentioned_tickers = EXCLUDED.mentioned_tickers,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        r["chunk_id"], r["doc_id"], r.get("chunk_index"), r["chunk_text"],
                        r.get("chunk_char_start"), r.get("chunk_char_end"), r.get("chunk_token_estimate"),
                        r.get("document_year"), r.get("document_year_source"), dumps(r.get("mentioned_years", [])),
                        dumps(r.get("mentioned_tickers", [])), r.get("source_name"), r.get("source_type"),
                        r.get("title"), r.get("chunk_quality_status"), dumps(r.get("metadata", {})),
                        vector_literal(r.get("embedding", [])),
                    ),
                )
                counts["evidence_chunks"] += 1
            for r in runs:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.content_build_runs (
                        run_id, input_path, output_dir, total_lines, valid_raw_documents, skipped_lines,
                        normalized_documents, documents_with_year, documents_without_year,
                        documents_with_entity, total_chunks, chunks_with_year, chunks_with_entity, notes
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        input_path = EXCLUDED.input_path,
                        output_dir = EXCLUDED.output_dir,
                        total_lines = EXCLUDED.total_lines,
                        valid_raw_documents = EXCLUDED.valid_raw_documents,
                        skipped_lines = EXCLUDED.skipped_lines,
                        normalized_documents = EXCLUDED.normalized_documents,
                        documents_with_year = EXCLUDED.documents_with_year,
                        documents_without_year = EXCLUDED.documents_without_year,
                        documents_with_entity = EXCLUDED.documents_with_entity,
                        total_chunks = EXCLUDED.total_chunks,
                        chunks_with_year = EXCLUDED.chunks_with_year,
                        chunks_with_entity = EXCLUDED.chunks_with_entity,
                        notes = EXCLUDED.notes
                    """,
                    (
                        r["run_id"], r.get("input_path"), r.get("output_dir"), r.get("total_lines"),
                        r.get("valid_raw_documents"), r.get("skipped_lines"), r.get("normalized_documents"),
                        r.get("documents_with_year"), r.get("documents_without_year"), r.get("documents_with_entity"),
                        r.get("total_chunks"), r.get("chunks_with_year"), r.get("chunks_with_entity"), r.get("notes"),
                    ),
                )
                counts["content_build_runs"] += 1

            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in retrieval_rows:
                if row.get("retrieval_run_id") and row.get("chunk_id"):
                    grouped[row["retrieval_run_id"]].append(row)
            if retrieval_rows and not grouped:
                print("Warning: content_retrieval_demo_results.csv lacks retrieval_run_id/chunk_id; skipping retrieval runs")
            for run_id, rows in grouped.items():
                first = rows[0]
                chunk_ids = [row["chunk_id"] for row in rows]
                years = sorted({parse_int(row.get("document_year")) for row in rows if parse_int(row.get("document_year")) is not None})
                tickers = sorted({
                    ticker
                    for row in rows
                    for ticker in (parse_jsonish(row.get("mentioned_tickers")) or [])
                })
                cur.execute(
                    f"""
                    INSERT INTO {schema}.content_retrieval_runs (
                        retrieval_run_id, query_text, filter_year_start, filter_year_end, filter_ticker,
                        top_k, retrieved_chunk_ids, retrieved_years, retrieved_tickers
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
                    ON CONFLICT (retrieval_run_id) DO UPDATE SET
                        query_text = EXCLUDED.query_text,
                        top_k = EXCLUDED.top_k,
                        retrieved_chunk_ids = EXCLUDED.retrieved_chunk_ids,
                        retrieved_years = EXCLUDED.retrieved_years,
                        retrieved_tickers = EXCLUDED.retrieved_tickers
                    """,
                    (
                        run_id, first.get("query_text"), None, None, None, len(rows),
                        dumps(chunk_ids), dumps(years), dumps(tickers),
                    ),
                )
                counts["content_retrieval_runs"] += 1
        conn.commit()

    for table, count in counts.items():
        print(f"{args.schema}.{table}: upserted {count}")


if __name__ == "__main__":
    main()
