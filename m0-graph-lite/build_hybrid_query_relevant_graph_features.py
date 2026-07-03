#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

from db_utils import add_db_args, connect_db, ensure_output_dirs, execute_sql_file, fetch_dataframe, write_dataframe_csv
from hybrid_relevance_utils import compute_dense_similarity, compute_hybrid_relevance


DEFAULT_BUILD_VERSION = "graph_lite_v2_hybrid"


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


def load_dense_model(dense_mode: str, embedding_model: str):
    if dense_mode == "none":
        print("Dense mode disabled; E5b will behave like lexical-enhanced baseline.")
        return None, "none", "none"
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedding_model)
        return model, "sentence_transformer", embedding_model
    except Exception as exc:
        print(f"WARNING: failed to load sentence-transformers model '{embedding_model}': {exc}")
        print("Dense mode disabled; E5b will behave like lexical-enhanced baseline.")
        return None, "none", "none"


def apply_schema(conn) -> None:
    path = Path(__file__).resolve().parent / "sql" / "005_create_hybrid_query_relevant_graph_tables.sql"
    execute_sql_file(conn, path)
    print(f"applied {path}")


def clear_existing(conn, build_version: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM m0_graph.hybrid_candidate_query_graph_features WHERE build_version = %s", (build_version,))
        cur.execute("DELETE FROM m0_graph.hybrid_query_year_evidence WHERE build_version = %s", (build_version,))
        cur.execute("DELETE FROM m0_graph.hybrid_query_relevant_chunks WHERE build_version = %s", (build_version,))
    conn.commit()


def selected_inputs(conn, method: str, setting: str, limit: int) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT ar.agent_run_id, ar.m0_input_id, ar.bridge_id, br.ticker,
                   br.rewritten_query AS query_text, br.rewrite_type
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
        cur.execute("SELECT * FROM runtime.m0_candidate_windows WHERE candidate_window_id LIKE %s ORDER BY candidate_window_id", (f"{agent_run_id}__cand_%",))
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
            WHERE ec.document_year = %s AND de.ticker = %s
            ORDER BY ec.chunk_id
            LIMIT %s
            """,
            (year, ticker, max_chunks),
        )
        return [dict(r) for r in cur.fetchall()]


def embed_text(model, text: str, cache: dict[str, Any]):
    if model is None:
        return None
    key = text or ""
    if key not in cache:
        cache[key] = model.encode(key, normalize_embeddings=True)
    return cache[key]


def insert_hybrid_chunk(conn, item: dict[str, Any], year: int, chunk: dict[str, Any], score: dict[str, Any], rank: int, dense_model: str, build_version: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO m0_graph.hybrid_query_relevant_chunks (
                m0_input_id, bridge_id, ticker, query_text, rewrite_type, year, chunk_id,
                doc_id, chunk_text, lexical_score, dense_similarity_score, signal_bonus,
                hybrid_relevance_score, ticker_match, market_signal_match, numeric_signal_match,
                year_match, rank_in_year, match_reason, dense_model_name, build_version
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (m0_input_id, year, chunk_id, build_version) DO UPDATE SET
                lexical_score = EXCLUDED.lexical_score,
                dense_similarity_score = EXCLUDED.dense_similarity_score,
                signal_bonus = EXCLUDED.signal_bonus,
                hybrid_relevance_score = EXCLUDED.hybrid_relevance_score,
                rank_in_year = EXCLUDED.rank_in_year,
                match_reason = EXCLUDED.match_reason
            """,
            (
                item["m0_input_id"], item["bridge_id"], item["ticker"], item["query_text"], item["rewrite_type"],
                year, chunk["chunk_id"], chunk["doc_id"], chunk["chunk_text"], score["lexical_score"],
                score["dense_similarity_score"], score["signal_bonus"], score["hybrid_relevance_score"],
                score["ticker_match"], score["market_signal_match"], score["numeric_signal_match"],
                int(chunk["document_year"]) == year, rank, score["match_reason"], dense_model, build_version,
            ),
        )


def insert_hybrid_year(conn, item: dict[str, Any], year: int, total_chunks: int, top_chunks: list[dict[str, Any]], top_k: int, dense_model: str, build_version: str) -> None:
    scores = [float(c["score"]["hybrid_relevance_score"]) for c in top_chunks]
    dense_scores = [float(c["score"]["dense_similarity_score"]) for c in top_chunks if c["score"]["dense_similarity_score"] is not None]
    lexical_scores = [float(c["score"]["lexical_score"]) for c in top_chunks]
    count = len(scores)
    max_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / count if count else 0.0
    sum_score = sum(scores)
    year_score = 0.50 * max_score + 0.30 * avg_score + 0.20 * min(1.0, count / max(1, top_k))
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO m0_graph.hybrid_query_year_evidence (
                m0_input_id, bridge_id, ticker, query_text, rewrite_type, year,
                total_candidate_chunks, hybrid_relevant_chunk_count, max_hybrid_relevance,
                avg_top_hybrid_relevance, sum_top_hybrid_relevance, avg_dense_similarity,
                avg_lexical_score, hybrid_year_relevance_score, year_has_hybrid_relevant_evidence,
                top_chunk_ids, top_chunk_scores, dense_model_name, build_version
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s)
            ON CONFLICT (m0_input_id, year, build_version) DO UPDATE SET
                total_candidate_chunks = EXCLUDED.total_candidate_chunks,
                hybrid_relevant_chunk_count = EXCLUDED.hybrid_relevant_chunk_count,
                max_hybrid_relevance = EXCLUDED.max_hybrid_relevance,
                avg_top_hybrid_relevance = EXCLUDED.avg_top_hybrid_relevance,
                sum_top_hybrid_relevance = EXCLUDED.sum_top_hybrid_relevance,
                avg_dense_similarity = EXCLUDED.avg_dense_similarity,
                avg_lexical_score = EXCLUDED.avg_lexical_score,
                hybrid_year_relevance_score = EXCLUDED.hybrid_year_relevance_score,
                year_has_hybrid_relevant_evidence = EXCLUDED.year_has_hybrid_relevant_evidence,
                top_chunk_ids = EXCLUDED.top_chunk_ids,
                top_chunk_scores = EXCLUDED.top_chunk_scores
            """,
            (
                item["m0_input_id"], item["bridge_id"], item["ticker"], item["query_text"], item["rewrite_type"],
                year, total_chunks, count, max_score, avg_score, sum_score,
                sum(dense_scores) / len(dense_scores) if dense_scores else None,
                sum(lexical_scores) / len(lexical_scores) if lexical_scores else 0.0,
                year_score, year_score >= 0.20, dump([c["chunk"]["chunk_id"] for c in top_chunks]),
                dump(scores), dense_model, build_version,
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


def balance(scores: list[float]) -> float:
    if not scores or max(scores) <= 0:
        return 0.0
    mean = sum(scores) / len(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return 1 / (1 + (std / max(mean, 0.001)))


def fetch_base_graph_feature(conn, candidate_window_id: str) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM m0_graph.candidate_graph_features WHERE candidate_window_id = %s ORDER BY created_at DESC LIMIT 1", (candidate_window_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def fetch_hybrid_years(conn, m0_input_id: str, build_version: str, years: list[int]) -> dict[int, dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM m0_graph.hybrid_query_year_evidence WHERE m0_input_id = %s AND build_version = %s AND year = ANY(%s)", (m0_input_id, build_version, years))
        return {int(r["year"]): dict(r) for r in cur.fetchall()}


def build_hybrid_years(conn, item: dict[str, Any], years: list[int], top_k: int, max_chunks: int, build_version: str, model, dense_model: str, embedding_cache: dict[str, Any]) -> None:
    query_embedding = embed_text(model, item["query_text"], embedding_cache)
    for year in years:
        chunks = fetch_chunks_for_year(conn, item["ticker"], year, max_chunks)
        scored = []
        for chunk in chunks:
            dense = None
            if query_embedding is not None:
                chunk_embedding = embed_text(model, chunk["chunk_text"], embedding_cache)
                dense = compute_dense_similarity(query_embedding, chunk_embedding)
            score = compute_hybrid_relevance(item["query_text"], chunk["chunk_text"], item["ticker"], dense)
            if score["hybrid_relevance_score"] > 0:
                scored.append({"chunk": chunk, "score": score})
        scored.sort(key=lambda x: x["score"]["hybrid_relevance_score"], reverse=True)
        top = scored[:top_k]
        for rank, entry in enumerate(top, start=1):
            insert_hybrid_chunk(conn, item, year, entry["chunk"], entry["score"], rank, dense_model, build_version)
        insert_hybrid_year(conn, item, year, len(chunks), top, top_k, dense_model, build_version)


def insert_hybrid_candidate_features(conn, item: dict[str, Any], candidate: dict[str, Any], year_rows: dict[int, dict[str, Any]], dense_model: str, build_version: str) -> None:
    years = list(range(int(candidate["start_year"]), int(candidate["end_year"]) + 1))
    scores = [float(year_rows.get(year, {}).get("hybrid_year_relevance_score") or 0.0) for year in years]
    flags = [bool(year_rows.get(year, {}).get("year_has_hybrid_relevant_evidence")) for year in years]
    length = len(years)
    relevant_count = sum(1 for flag in flags if flag)
    gap_count, max_gap = gap_stats(flags)
    total = sum(scores)
    half = max(1, length // 2)
    early_mass = sum(scores[:half]) / total if total > 0 else 0.0
    late_mass = sum(scores[half:]) / total if total > 0 else 0.0
    base = fetch_base_graph_feature(conn, candidate["candidate_window_id"]) or {}
    prefix_support = 0.5 * float(base.get("prefix_support_score") or 0.5) + 0.5 * early_mass
    suffix_support = 0.5 * float(base.get("suffix_support_score") or 0.5) + 0.5 * late_mass
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
        directional = 0.5 * (relevant_count / max(1, length)) + 0.5 * balance(scores)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO m0_graph.hybrid_candidate_query_graph_features (
                candidate_window_id, m0_input_id, bridge_id, ticker, query_text, rewrite_type,
                start_year, end_year, window_type, window_length, hybrid_relevant_year_count,
                hybrid_relevant_year_ratio, hybrid_window_relevance_mean, hybrid_window_relevance_max,
                hybrid_window_relevance_min, hybrid_window_relevance_sum, hybrid_window_relevance_balance,
                hybrid_semantic_gap_count, max_hybrid_semantic_gap_length, prefix_hybrid_query_support,
                suffix_hybrid_query_support, normalized_hybrid_relevance_center,
                hybrid_directional_relevance_fit, dense_model_name, build_version
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (candidate_window_id, build_version) DO UPDATE SET
                hybrid_relevant_year_count = EXCLUDED.hybrid_relevant_year_count,
                hybrid_relevant_year_ratio = EXCLUDED.hybrid_relevant_year_ratio,
                hybrid_window_relevance_mean = EXCLUDED.hybrid_window_relevance_mean,
                hybrid_window_relevance_balance = EXCLUDED.hybrid_window_relevance_balance,
                hybrid_semantic_gap_count = EXCLUDED.hybrid_semantic_gap_count,
                max_hybrid_semantic_gap_length = EXCLUDED.max_hybrid_semantic_gap_length,
                hybrid_directional_relevance_fit = EXCLUDED.hybrid_directional_relevance_fit
            """,
            (
                candidate["candidate_window_id"], item["m0_input_id"], item["bridge_id"], item["ticker"],
                item["query_text"], item["rewrite_type"], candidate["start_year"], candidate["end_year"],
                candidate["window_type"], length, relevant_count, relevant_count / max(1, length),
                total / max(1, length), max(scores) if scores else 0.0, min(scores) if scores else 0.0,
                total, balance(scores), gap_count, max_gap, prefix_support, suffix_support,
                max(0.0, min(1.0, relevance_center)), directional, dense_model, build_version,
            ),
        )


def build_features(conn, inputs: list[dict[str, Any]], top_k: int, max_chunks: int, build_version: str, model, dense_model: str) -> None:
    embedding_cache: dict[str, Any] = {}
    for item in inputs:
        candidates = fetch_candidates(conn, item["agent_run_id"])
        if not candidates:
            continue
        min_year = min(int(c["start_year"]) for c in candidates)
        max_year = max(int(c["end_year"]) for c in candidates)
        years = list(range(min_year, max_year + 1))
        build_hybrid_years(conn, item, years, top_k, max_chunks, build_version, model, dense_model, embedding_cache)
        year_rows = fetch_hybrid_years(conn, item["m0_input_id"], build_version, years)
        for candidate in candidates:
            insert_hybrid_candidate_features(conn, item, candidate, year_rows, dense_model, build_version)
        conn.commit()


def export_outputs(conn) -> None:
    exports = {
        "hybrid_query_relevant_chunks.csv": "SELECT * FROM m0_graph.hybrid_query_relevant_chunks ORDER BY m0_input_id, year, rank_in_year",
        "hybrid_query_year_evidence.csv": "SELECT * FROM m0_graph.hybrid_query_year_evidence ORDER BY m0_input_id, year",
        "hybrid_candidate_query_graph_features.csv": "SELECT * FROM m0_graph.hybrid_candidate_query_graph_features ORDER BY m0_input_id, candidate_window_id",
        "hybrid_candidate_query_graph_feature_summary.csv": "SELECT * FROM m0_graph.v_hybrid_candidate_query_graph_feature_summary ORDER BY rewrite_type, window_type",
    }
    for name, sql in exports.items():
        write_dataframe_csv(fetch_dataframe(conn, sql), Path("m0-graph-lite/outputs/csv") / name)


def md_table(df, max_rows=16) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).fillna("")
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c])[:120] for c in view.columns) + " |")
    return "\n".join(lines)


def generate_report(conn, output: Path, dense_mode: str, dense_model: str) -> None:
    year_summary = fetch_dataframe(conn, "SELECT * FROM m0_graph.v_hybrid_query_year_evidence_summary ORDER BY rewrite_type, build_version")
    feature_summary = fetch_dataframe(conn, "SELECT * FROM m0_graph.v_hybrid_candidate_query_graph_feature_summary ORDER BY rewrite_type, window_type")
    lines = [
        "# Hybrid Query-Relevant Graph-Lite Report",
        "",
        "## Why E5b Was Added",
        "E5 lexical relevance may be too weak. E5b upgrades chunk relevance to hybrid lexical + dense semantic relevance plus ticker/market/numeric signals.",
        "",
        "## Method",
        "lexical + dense semantic + signal bonus -> hybrid chunk relevance -> hybrid year relevance -> hybrid candidate-window relevance",
        "",
        f"- dense_mode used: `{dense_mode}`",
        f"- embedding model used: `{dense_model}`",
        "",
        "## Feature Definitions",
        "- hybrid_relevance_score combines lexical_score, dense_similarity_score, and signal_bonus.",
        "- hybrid_year_relevance_score aggregates top chunk relevance per year.",
        "- hybrid_relevant_year_ratio = hybrid-relevant years / window length.",
        "- hybrid_window_relevance_balance = 1 / (1 + coefficient of variation over yearly scores).",
        "- hybrid_semantic_gap_count counts contiguous years without hybrid-relevant evidence.",
        "- prefix/suffix_hybrid_query_support capture early/late relevance mass.",
        "- hybrid_directional_relevance_fit maps relative intent to prefix/suffix/broad evidence.",
        "",
        "## Data Leakage Policy",
        "Gold labels are not used in feature construction.",
        "",
        "## Hybrid Year Evidence Summary",
        md_table(year_summary),
        "",
        "## Hybrid Candidate Feature Summary",
        md_table(feature_summary),
        "",
        "## Interpretation",
    ]
    if not feature_summary.empty and feature_summary["avg_hybrid_relevant_year_ratio"].std() > 0:
        lines.append("- Hybrid features vary; proceed to E5b hybrid query-relevant graph scoring.")
    else:
        lines.append("- Hybrid features are fairly flat; future work needs year-level answerability or trend-specific evidence scoring.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report -> {output}")


def print_summary(conn, dense_mode: str, dense_model: str) -> None:
    df = fetch_dataframe(conn, """
        SELECT
            (SELECT COUNT(DISTINCT m0_input_id) FROM m0_graph.hybrid_query_year_evidence) AS inputs,
            (SELECT COUNT(*) FROM m0_graph.hybrid_query_relevant_chunks) AS chunks,
            (SELECT COUNT(*) FROM m0_graph.hybrid_query_year_evidence) AS years,
            COUNT(*) AS candidates,
            AVG(hybrid_relevant_year_ratio) AS ratio,
            AVG(hybrid_window_relevance_mean) AS relevance,
            AVG(hybrid_semantic_gap_count) AS gaps,
            AVG(hybrid_directional_relevance_fit) AS directional
        FROM m0_graph.hybrid_candidate_query_graph_features
    """)
    row = df.iloc[0]
    print(f"number of m0 inputs processed: {int(row['inputs'] or 0)}")
    print(f"number of hybrid query-relevant chunks stored: {int(row['chunks'] or 0)}")
    print(f"number of hybrid query-year evidence rows: {int(row['years'] or 0)}")
    print(f"number of hybrid candidate query graph feature rows: {int(row['candidates'] or 0)}")
    print(f"avg hybrid_relevant_year_ratio: {float(row['ratio'] or 0):.4f}")
    print(f"avg hybrid_window_relevance_mean: {float(row['relevance'] or 0):.4f}")
    print(f"avg hybrid_semantic_gap_count: {float(row['gaps'] or 0):.4f}")
    print(f"avg hybrid_directional_relevance_fit: {float(row['directional'] or 0):.4f}")
    print(f"dense_mode actually used: {dense_mode}")
    print(f"embedding model actually used: {dense_model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_db_args(parser)
    parser.add_argument("--method", default="vector_probe_agent_v1")
    parser.add_argument("--m0-input-setting", default="query_with_corpus_metadata")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k-per-year", type=int, default=5)
    parser.add_argument("--max-chunks-per-year", type=int, default=50)
    parser.add_argument("--build-version", default=DEFAULT_BUILD_VERSION)
    parser.add_argument("--dense-mode", choices=["none", "sentence_transformer"], default="sentence_transformer")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    model, dense_mode, dense_model = load_dense_model(args.dense_mode, args.embedding_model)
    with connect_db(args) as conn:
        apply_schema(conn)
        if args.clear_existing:
            clear_existing(conn, args.build_version)
        inputs = selected_inputs(conn, args.method, args.m0_input_setting, args.limit)
        build_features(conn, inputs, args.top_k_per_year, args.max_chunks_per_year, args.build_version, model, dense_model)
        export_outputs(conn)
        generate_report(conn, Path("m0-graph-lite/outputs/markdown/hybrid_query_relevant_graph_lite_report.md"), dense_mode, dense_model)
        print_summary(conn, dense_mode, dense_model)


if __name__ == "__main__":
    main()
