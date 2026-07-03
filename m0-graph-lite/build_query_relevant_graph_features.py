#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

from db_utils import add_db_args, connect_db, ensure_output_dirs, execute_sql_file, fetch_dataframe, write_dataframe_csv
from query_relevance_utils import compute_lexical_relevance


DEFAULT_BUILD_VERSION = "graph_lite_v2_query_relevant"


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def detect_relative_intent(query_text: str, rewrite_type: str | None = None) -> str:
    q = (query_text or "").lower()
    if any(p in q for p in ("earlier period", "earlier periods", "early period", "early years", "earlier years", "before", "prior to", "previously", "initial period", "beginning")):
        return "earlier"
    if any(p in q for p in ("recent years", "most recent", "recently", "latest", "current period")):
        return "recent"
    if any(p in q for p in ("later period", "later periods", "later years", "after", "following", "subsequent", "toward the end", "by the end")):
        return "later"
    if any(p in q for p in ("over time", "overall", "generally trend", "general trend", "multi-year", "across years", "throughout the period", "long-term", "historical trend")):
        return "broad"
    if rewrite_type == "A":
        return "broad_unknown"
    return "unknown"


def apply_schema(conn) -> None:
    path = Path(__file__).resolve().parent / "sql" / "004_create_query_relevant_graph_tables.sql"
    execute_sql_file(conn, path)
    print(f"applied {path}")


def clear_existing(conn, build_version: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM m0_graph.candidate_query_graph_features WHERE build_version = %s", (build_version,))
        cur.execute("DELETE FROM m0_graph.query_year_evidence WHERE build_version = %s", (build_version,))
        cur.execute("DELETE FROM m0_graph.query_relevant_chunks WHERE build_version = %s", (build_version,))
    conn.commit()


def selected_inputs(conn, method: str, setting: str, limit: int) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
                ar.agent_run_id,
                ar.m0_input_id,
                ar.bridge_id,
                br.ticker,
                br.rewritten_query AS query_text,
                br.rewrite_type
            FROM runtime.m0_agent_runs ar
            JOIN qa.bridge_records br ON br.bridge_id = ar.bridge_id
            WHERE ar.method = %s AND ar.m0_input_setting = %s
            ORDER BY ar.created_at DESC
            LIMIT %s
            """,
            (method, setting, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_candidates(conn, agent_run_id: str) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT *
            FROM runtime.m0_candidate_windows
            WHERE candidate_window_id LIKE %s
            ORDER BY candidate_window_id
            """,
            (f"{agent_run_id}__cand_%",),
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_chunks_for_year(conn, ticker: str, year: int, max_chunks: int) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT chunk_id, doc_id, chunk_text, document_year
            FROM content.evidence_chunks
            WHERE document_year = %s
              AND jsonb_typeof(mentioned_tickers) = 'array'
              AND mentioned_tickers ? %s
            ORDER BY chunk_id
            LIMIT %s
            """,
            (year, ticker, max_chunks),
        )
        rows = [dict(r) for r in cur.fetchall()]
        if rows:
            return rows
        cur.execute(
            """
            SELECT DISTINCT ec.chunk_id, ec.doc_id, ec.chunk_text, ec.document_year
            FROM content.evidence_chunks ec
            JOIN content.document_entities de ON de.doc_id = ec.doc_id
            WHERE ec.document_year = %s
              AND de.ticker = %s
            ORDER BY ec.chunk_id
            LIMIT %s
            """,
            (year, ticker, max_chunks),
        )
        return [dict(r) for r in cur.fetchall()]


def insert_query_relevant_chunk(conn, item: dict[str, Any], year: int, chunk: dict[str, Any], score: dict[str, Any], rank: int, build_version: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO m0_graph.query_relevant_chunks (
                m0_input_id, bridge_id, ticker, query_text, rewrite_type, year, chunk_id,
                doc_id, chunk_text, relevance_score, lexical_score, ticker_match, year_match,
                rank_in_year, match_reason, build_version
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (m0_input_id, year, chunk_id, build_version) DO UPDATE SET
                relevance_score = EXCLUDED.relevance_score,
                lexical_score = EXCLUDED.lexical_score,
                rank_in_year = EXCLUDED.rank_in_year,
                match_reason = EXCLUDED.match_reason
            """,
            (
                item["m0_input_id"], item["bridge_id"], item["ticker"], item["query_text"],
                item["rewrite_type"], year, chunk["chunk_id"], chunk["doc_id"], chunk["chunk_text"],
                score["relevance_score"], score["lexical_score"], score["ticker_match"],
                int(chunk["document_year"]) == year, rank, score["match_reason"], build_version,
            ),
        )


def insert_query_year_evidence(conn, item: dict[str, Any], year: int, total_chunks: int, top_chunks: list[dict[str, Any]], top_k: int, build_version: str) -> None:
    scores = [float(c["score"]["relevance_score"]) for c in top_chunks]
    max_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    sum_score = sum(scores)
    count = len(scores)
    year_score = 0.50 * max_score + 0.30 * avg_score + 0.20 * min(1.0, count / max(1, top_k))
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO m0_graph.query_year_evidence (
                m0_input_id, bridge_id, ticker, query_text, rewrite_type, year,
                total_candidate_chunks, query_relevant_chunk_count, max_chunk_relevance,
                avg_top_chunk_relevance, sum_top_chunk_relevance, year_relevance_score,
                year_has_relevant_evidence, top_chunk_ids, top_chunk_scores, build_version
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
            ON CONFLICT (m0_input_id, year, build_version) DO UPDATE SET
                total_candidate_chunks = EXCLUDED.total_candidate_chunks,
                query_relevant_chunk_count = EXCLUDED.query_relevant_chunk_count,
                max_chunk_relevance = EXCLUDED.max_chunk_relevance,
                avg_top_chunk_relevance = EXCLUDED.avg_top_chunk_relevance,
                sum_top_chunk_relevance = EXCLUDED.sum_top_chunk_relevance,
                year_relevance_score = EXCLUDED.year_relevance_score,
                year_has_relevant_evidence = EXCLUDED.year_has_relevant_evidence,
                top_chunk_ids = EXCLUDED.top_chunk_ids,
                top_chunk_scores = EXCLUDED.top_chunk_scores
            """,
            (
                item["m0_input_id"], item["bridge_id"], item["ticker"], item["query_text"],
                item["rewrite_type"], year, total_chunks, count, max_score, avg_score, sum_score,
                year_score, year_score >= 0.20,
                dump([c["chunk"]["chunk_id"] for c in top_chunks]),
                dump(scores),
                build_version,
            ),
        )


def gap_stats(flags: list[bool]) -> tuple[int, int]:
    count = 0
    max_gap = 0
    current = 0
    for flag in flags:
        if flag:
            if current:
                count += 1
                max_gap = max(max_gap, current)
                current = 0
        else:
            current += 1
    if current:
        count += 1
        max_gap = max(max_gap, current)
    return count, max_gap


def relevance_balance(scores: list[float]) -> float:
    if not scores or max(scores) <= 0:
        return 0.0
    mean = sum(scores) / len(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return 1 / (1 + (std / max(mean, 0.001)))


def build_query_years(conn, item: dict[str, Any], years: list[int], top_k: int, max_chunks: int, build_version: str) -> None:
    for year in years:
        chunks = fetch_chunks_for_year(conn, item["ticker"], year, max_chunks)
        scored = []
        for chunk in chunks:
            score = compute_lexical_relevance(item["query_text"], chunk["chunk_text"], item["ticker"])
            if score["relevance_score"] > 0:
                scored.append({"chunk": chunk, "score": score})
        scored.sort(key=lambda x: x["score"]["relevance_score"], reverse=True)
        top = scored[:top_k]
        for rank, entry in enumerate(top, start=1):
            insert_query_relevant_chunk(conn, item, year, entry["chunk"], entry["score"], rank, build_version)
        insert_query_year_evidence(conn, item, year, len(chunks), top, top_k, build_version)


def fetch_year_evidence(conn, m0_input_id: str, build_version: str, years: list[int]) -> dict[int, dict[str, Any]]:
    if not years:
        return {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT *
            FROM m0_graph.query_year_evidence
            WHERE m0_input_id = %s
              AND build_version = %s
              AND year = ANY(%s)
            """,
            (m0_input_id, build_version, years),
        )
        return {int(r["year"]): dict(r) for r in cur.fetchall()}


def fetch_base_graph_feature(conn, candidate_window_id: str) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT *
            FROM m0_graph.candidate_graph_features
            WHERE candidate_window_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (candidate_window_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def insert_candidate_query_features(conn, item: dict[str, Any], candidate: dict[str, Any], year_rows: dict[int, dict[str, Any]], build_version: str) -> None:
    years = list(range(int(candidate["start_year"]), int(candidate["end_year"]) + 1))
    scores = [float(year_rows.get(year, {}).get("year_relevance_score") or 0.0) for year in years]
    flags = [bool(year_rows.get(year, {}).get("year_has_relevant_evidence")) for year in years]
    length = len(years)
    relevant_count = sum(1 for flag in flags if flag)
    gap_count, max_gap = gap_stats(flags)
    balance = relevance_balance(scores)
    total = sum(scores)
    half = max(1, length // 2)
    early_mass = sum(scores[:half]) / total if total > 0 else 0.0
    late_mass = sum(scores[half:]) / total if total > 0 else 0.0
    base = fetch_base_graph_feature(conn, candidate["candidate_window_id"]) or {}
    prefix_base = float(base.get("prefix_support_score") or 0.5)
    suffix_base = float(base.get("suffix_support_score") or 0.5)
    prefix_support = 0.5 * prefix_base + 0.5 * early_mass
    suffix_support = 0.5 * suffix_base + 0.5 * late_mass
    timeline_min = int(base.get("timeline_min_year") or min(years))
    timeline_len = max(1, int(base.get("timeline_length") or (max(years) - timeline_min + 1)))
    denom = max(1, timeline_len - 1)
    if total > 0:
        weighted_year = sum(year * score for year, score in zip(years, scores)) / total
        relevance_center = (weighted_year - timeline_min) / denom
    else:
        relevance_center = float(base.get("normalized_center") or 0.5)
    intent = detect_relative_intent(item["query_text"], item["rewrite_type"])
    if intent == "earlier":
        directional = prefix_support
    elif intent in {"later", "recent"}:
        directional = suffix_support
    else:
        directional = 0.5 * (relevant_count / max(1, length)) + 0.5 * balance
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO m0_graph.candidate_query_graph_features (
                candidate_window_id, m0_input_id, bridge_id, ticker, query_text, rewrite_type,
                start_year, end_year, window_type, window_length, query_relevant_year_count,
                query_relevant_year_ratio, window_relevance_mean, window_relevance_max,
                window_relevance_min, window_relevance_sum, window_relevance_balance,
                semantic_gap_count, max_semantic_gap_length, prefix_query_support,
                suffix_query_support, normalized_query_relevance_center,
                directional_relevance_fit, build_version
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (candidate_window_id, build_version) DO UPDATE SET
                query_relevant_year_count = EXCLUDED.query_relevant_year_count,
                query_relevant_year_ratio = EXCLUDED.query_relevant_year_ratio,
                window_relevance_mean = EXCLUDED.window_relevance_mean,
                window_relevance_balance = EXCLUDED.window_relevance_balance,
                semantic_gap_count = EXCLUDED.semantic_gap_count,
                max_semantic_gap_length = EXCLUDED.max_semantic_gap_length,
                directional_relevance_fit = EXCLUDED.directional_relevance_fit
            """,
            (
                candidate["candidate_window_id"], item["m0_input_id"], item["bridge_id"], item["ticker"],
                item["query_text"], item["rewrite_type"], candidate["start_year"], candidate["end_year"],
                candidate["window_type"], length, relevant_count, relevant_count / max(1, length),
                sum(scores) / max(1, length), max(scores) if scores else 0.0, min(scores) if scores else 0.0,
                total, balance, gap_count, max_gap, prefix_support, suffix_support,
                max(0.0, min(1.0, relevance_center)), directional, build_version,
            ),
        )


def build_features(conn, inputs: list[dict[str, Any]], top_k: int, max_chunks: int, build_version: str) -> None:
    for item in inputs:
        candidates = fetch_candidates(conn, item["agent_run_id"])
        if not candidates:
            continue
        min_year = min(int(c["start_year"]) for c in candidates)
        max_year = max(int(c["end_year"]) for c in candidates)
        years = list(range(min_year, max_year + 1))
        build_query_years(conn, item, years, top_k, max_chunks, build_version)
        year_rows = fetch_year_evidence(conn, item["m0_input_id"], build_version, years)
        for candidate in candidates:
            insert_candidate_query_features(conn, item, candidate, year_rows, build_version)
        conn.commit()


def export_outputs(conn) -> None:
    exports = {
        "query_relevant_chunks.csv": "SELECT * FROM m0_graph.query_relevant_chunks ORDER BY m0_input_id, year, rank_in_year",
        "query_year_evidence.csv": "SELECT * FROM m0_graph.query_year_evidence ORDER BY m0_input_id, year",
        "candidate_query_graph_features.csv": "SELECT * FROM m0_graph.candidate_query_graph_features ORDER BY m0_input_id, candidate_window_id",
        "candidate_query_graph_feature_summary.csv": "SELECT * FROM m0_graph.v_candidate_query_graph_feature_summary ORDER BY rewrite_type, window_type",
    }
    for name, sql in exports.items():
        write_dataframe_csv(fetch_dataframe(conn, sql), Path("m0-graph-lite/outputs/csv") / name)


def generate_query_report(conn, output: Path) -> None:
    year_summary = fetch_dataframe(conn, "SELECT * FROM m0_graph.v_query_year_evidence_summary ORDER BY rewrite_type, build_version")
    feature_summary = fetch_dataframe(conn, "SELECT * FROM m0_graph.v_candidate_query_graph_feature_summary ORDER BY rewrite_type, window_type")
    def table(df, max_rows=16):
        if df.empty:
            return "_No rows._"
        view = df.head(max_rows).fillna("")
        lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c])[:120] for c in view.columns) + " |")
        return "\n".join(lines)
    lines = [
        "# Query-Relevant Graph-Lite Report",
        "",
        "## Why V2 Was Added",
        "Graph-lite v1 modeled content availability. E4 showed count-based graph-lite was insufficient because the content timeline was too dense. V2 models query-relevant evidence by aggregating chunk relevance into years and candidate windows.",
        "",
        "## Method",
        "chunk relevance -> year relevance -> candidate window relevance",
        "",
        "## Feature Definitions",
        "- chunk_relevance_score: lexical query/chunk relevance with ticker, phrase, numeric, and market bonuses.",
        "- year_relevance_score = 0.50 * max_chunk_relevance + 0.30 * avg_top_chunk_relevance + 0.20 * top-k coverage.",
        "- query_relevant_year_ratio = relevant years / window length.",
        "- window_relevance_mean / balance summarize relevance across years.",
        "- semantic_gap_count counts contiguous years without query-relevant evidence.",
        "- prefix_query_support and suffix_query_support combine timeline position with early/late relevance mass.",
        "- directional_relevance_fit maps query intent onto prefix/suffix/broad relevance.",
        "",
        "## Data Leakage Policy",
        "Gold labels are not used in feature construction.",
        "",
        "## Query Year Evidence Summary",
        table(year_summary),
        "",
        "## Candidate Query Graph Feature Summary",
        table(feature_summary),
        "",
        "## Interpretation",
    ]
    if not feature_summary.empty and feature_summary["avg_query_relevant_year_ratio"].std() > 0:
        lines.append("- Query-relevant features vary; proceed to E5 query-relevant graph scoring.")
    else:
        lines.append("- Features are fairly flat; lexical relevance may be too weak and embedding relevance may be needed later.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report -> {output}")


def print_summary(conn) -> None:
    df = fetch_dataframe(conn, """
        SELECT
            (SELECT COUNT(DISTINCT m0_input_id) FROM m0_graph.query_year_evidence) AS inputs,
            (SELECT COUNT(*) FROM m0_graph.query_relevant_chunks) AS chunks,
            (SELECT COUNT(*) FROM m0_graph.query_year_evidence) AS years,
            COUNT(*) AS candidates,
            AVG(query_relevant_year_ratio) AS ratio,
            AVG(window_relevance_mean) AS relevance,
            AVG(semantic_gap_count) AS gaps,
            AVG(directional_relevance_fit) AS directional
        FROM m0_graph.candidate_query_graph_features
    """)
    row = df.iloc[0]
    print(f"number of m0 inputs processed: {int(row['inputs'] or 0)}")
    print(f"number of query-relevant chunks stored: {int(row['chunks'] or 0)}")
    print(f"number of query-year evidence rows: {int(row['years'] or 0)}")
    print(f"number of candidate query graph feature rows: {int(row['candidates'] or 0)}")
    print(f"avg query_relevant_year_ratio: {float(row['ratio'] or 0):.4f}")
    print(f"avg window_relevance_mean: {float(row['relevance'] or 0):.4f}")
    print(f"avg semantic_gap_count: {float(row['gaps'] or 0):.4f}")
    print(f"avg directional_relevance_fit: {float(row['directional'] or 0):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_db_args(parser)
    parser.add_argument("--method", default="vector_probe_agent_v1")
    parser.add_argument("--m0-input-setting", default="query_with_corpus_metadata")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k-per-year", type=int, default=5)
    parser.add_argument("--max-chunks-per-year", type=int, default=50)
    parser.add_argument("--build-version", default=DEFAULT_BUILD_VERSION)
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    with connect_db(args) as conn:
        apply_schema(conn)
        if args.clear_existing:
            clear_existing(conn, args.build_version)
        inputs = selected_inputs(conn, args.method, args.m0_input_setting, args.limit)
        build_features(conn, inputs, args.top_k_per_year, args.max_chunks_per_year, args.build_version)
        export_outputs(conn)
        generate_query_report(conn, Path("m0-graph-lite/outputs/markdown/query_relevant_graph_lite_report.md"))
        print_summary(conn)


if __name__ == "__main__":
    main()
