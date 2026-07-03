from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

from db_utils import add_db_args, connect_db, ensure_output_dirs, execute_sql_file, fetch_dataframe, write_dataframe_csv
from export_graph_lite_report import generate_report
from graph_config import DEFAULT_BUILD_VERSION


def apply_schema(conn) -> None:
    base = Path(__file__).resolve().parent / "sql"
    for name in ["001_create_m0_graph_schema.sql", "002_create_m0_graph_tables.sql", "003_create_m0_graph_views.sql"]:
        execute_sql_file(conn, base / name)
        print(f"applied {base / name}")


def clear_existing(conn, build_version: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM m0_graph.candidate_graph_features WHERE graph_feature_version = %s", (build_version,))
        cur.execute("DELETE FROM m0_graph.candidate_year_coverage WHERE build_version = %s", (build_version,))
        cur.execute("DELETE FROM m0_graph.ticker_year_evidence WHERE build_version = %s", (build_version,))
    conn.commit()


def build_ticker_year_evidence(conn, build_version: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH expanded AS (
                SELECT
                    jsonb_array_elements_text(ec.mentioned_tickers) AS ticker,
                    ec.document_year AS year,
                    ec.chunk_id,
                    ec.doc_id
                FROM content.evidence_chunks ec
                WHERE ec.document_year IS NOT NULL
                  AND jsonb_typeof(ec.mentioned_tickers) = 'array'
                  AND jsonb_array_length(ec.mentioned_tickers) > 0
            ),
            counts AS (
                SELECT ticker, year, COUNT(DISTINCT doc_id) AS doc_count, COUNT(*) AS chunk_count
                FROM expanded
                GROUP BY ticker, year
            ),
            maxes AS (
                SELECT ticker, MAX(chunk_count) AS max_chunk_count
                FROM counts
                GROUP BY ticker
            )
            INSERT INTO m0_graph.ticker_year_evidence (
                ticker, year, doc_count, chunk_count, has_evidence, density_score, build_version
            )
            SELECT
                c.ticker, c.year, c.doc_count, c.chunk_count, true,
                CASE WHEN m.max_chunk_count > 0 THEN c.chunk_count::REAL / m.max_chunk_count ELSE 0 END,
                %s
            FROM counts c
            JOIN maxes m ON m.ticker = c.ticker
            ON CONFLICT (ticker, year, build_version) DO UPDATE SET
                doc_count = EXCLUDED.doc_count,
                chunk_count = EXCLUDED.chunk_count,
                has_evidence = EXCLUDED.has_evidence,
                density_score = EXCLUDED.density_score
            """,
            (build_version,),
        )
    conn.commit()


def selected_inputs(conn, method: str, setting: str, limit: int) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT agent_run_id, m0_input_id, bridge_id, ticker
            FROM runtime.m0_agent_runs
            WHERE method = %s AND m0_input_setting = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (method, setting, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_candidates(conn, agent_run_id: str) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT candidate_window_id, m0_input_id, bridge_id, ticker, start_year, end_year, window_type
            FROM runtime.m0_candidate_windows
            WHERE candidate_window_id LIKE %s
            ORDER BY candidate_window_id
            """,
            (f"{agent_run_id}__cand_%",),
        )
        return [dict(r) for r in cur.fetchall()]


def gap_stats(rows: list[dict[str, Any]]) -> tuple[int, int]:
    gap_count = 0
    max_gap = 0
    current = 0
    for row in rows:
        if row["has_evidence"]:
            if current:
                gap_count += 1
                max_gap = max(max_gap, current)
                current = 0
        else:
            current += 1
    if current:
        gap_count += 1
        max_gap = max(max_gap, current)
    return gap_count, max_gap


def build_candidate_features(conn, inputs: list[dict[str, Any]], build_version: str) -> None:
    with conn.cursor(row_factory=dict_row) as cur:
        for item in inputs:
            candidates = fetch_candidates(conn, item["agent_run_id"])
            if not candidates:
                continue
            timeline_min = min(int(c["start_year"]) for c in candidates)
            timeline_max = max(int(c["end_year"]) for c in candidates)
            timeline_len = timeline_max - timeline_min + 1
            for cand in candidates:
                years = list(range(int(cand["start_year"]), int(cand["end_year"]) + 1))
                coverage_rows = []
                for year in years:
                    cur.execute(
                        """
                        SELECT doc_count, chunk_count, has_evidence, density_score
                        FROM m0_graph.ticker_year_evidence
                        WHERE ticker = %s AND year = %s AND build_version = %s
                        """,
                        (cand["ticker"], year, build_version),
                    )
                    ev = cur.fetchone()
                    row = {
                        "year": year,
                        "doc_count": int(ev["doc_count"]) if ev else 0,
                        "chunk_count": int(ev["chunk_count"]) if ev else 0,
                        "has_evidence": bool(ev["has_evidence"]) if ev else False,
                        "density_score": float(ev["density_score"]) if ev else 0.0,
                    }
                    coverage_rows.append(row)
                    cur.execute(
                        """
                        INSERT INTO m0_graph.candidate_year_coverage (
                            candidate_window_id, m0_input_id, bridge_id, ticker, year,
                            start_year, end_year, window_type, has_evidence, chunk_count,
                            doc_count, is_gap, build_version
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (candidate_window_id, year, build_version) DO UPDATE SET
                            has_evidence = EXCLUDED.has_evidence,
                            chunk_count = EXCLUDED.chunk_count,
                            doc_count = EXCLUDED.doc_count,
                            is_gap = EXCLUDED.is_gap
                        """,
                        (
                            cand["candidate_window_id"], cand["m0_input_id"], cand["bridge_id"],
                            cand["ticker"], year, cand["start_year"], cand["end_year"], cand["window_type"],
                            row["has_evidence"], row["chunk_count"], row["doc_count"], not row["has_evidence"],
                            build_version,
                        ),
                    )
                length = len(years)
                center = (int(cand["start_year"]) + int(cand["end_year"])) / 2
                denom = max(1, timeline_len - 1)
                normalized_start = (int(cand["start_year"]) - timeline_min) / denom
                normalized_end = (int(cand["end_year"]) - timeline_min) / denom
                normalized_center = (center - timeline_min) / denom
                evidence_count = sum(1 for r in coverage_rows if r["has_evidence"])
                gap_count, max_gap = gap_stats(coverage_rows)
                chunk_counts = [r["chunk_count"] for r in coverage_rows]
                avg_chunks = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0
                if not chunk_counts or max(chunk_counts) == 0:
                    balance = 0.0
                else:
                    std = statistics.pstdev(chunk_counts) if len(chunk_counts) > 1 else 0.0
                    balance = 1 / (1 + (std / max(avg_chunks, 1)))
                cur.execute(
                    """
                    INSERT INTO m0_graph.candidate_graph_features (
                        candidate_window_id, m0_input_id, bridge_id, ticker, start_year, end_year,
                        window_type, window_length, timeline_min_year, timeline_max_year, timeline_length,
                        normalized_start, normalized_end, normalized_center, years_with_evidence_count,
                        year_continuity_score, gap_count, max_gap_length, prefix_support_score,
                        suffix_support_score, coverage_density_score, density_balance_score,
                        avg_chunk_count_per_year, max_chunk_count_in_year, min_chunk_count_in_year,
                        graph_feature_version
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (candidate_window_id, graph_feature_version) DO UPDATE SET
                        year_continuity_score = EXCLUDED.year_continuity_score,
                        gap_count = EXCLUDED.gap_count,
                        max_gap_length = EXCLUDED.max_gap_length,
                        density_balance_score = EXCLUDED.density_balance_score
                    """,
                    (
                        cand["candidate_window_id"], cand["m0_input_id"], cand["bridge_id"], cand["ticker"],
                        cand["start_year"], cand["end_year"], cand["window_type"], length,
                        timeline_min, timeline_max, timeline_len, normalized_start, normalized_end,
                        normalized_center, evidence_count, evidence_count / length if length else 0,
                        gap_count, max_gap, 1 - normalized_start, normalized_end,
                        sum(r["density_score"] for r in coverage_rows) / length if length else 0,
                        balance, avg_chunks, max(chunk_counts) if chunk_counts else 0,
                        min(chunk_counts) if chunk_counts else 0, build_version,
                    ),
                )
    conn.commit()


def export_outputs(conn) -> None:
    exports = {
        "ticker_year_evidence.csv": "SELECT * FROM m0_graph.ticker_year_evidence ORDER BY ticker, year",
        "candidate_year_coverage.csv": "SELECT * FROM m0_graph.candidate_year_coverage ORDER BY candidate_window_id, year",
        "candidate_graph_features.csv": "SELECT * FROM m0_graph.candidate_graph_features ORDER BY candidate_window_id",
        "candidate_graph_feature_summary.csv": "SELECT * FROM m0_graph.v_candidate_graph_feature_summary ORDER BY window_type",
    }
    for name, sql in exports.items():
        write_dataframe_csv(fetch_dataframe(conn, sql), Path("m0-graph-lite/outputs/csv") / name)


def print_summary(conn) -> None:
    df = fetch_dataframe(conn, "SELECT COUNT(DISTINCT ticker) tickers, COUNT(*) rows FROM m0_graph.ticker_year_evidence")
    feats = fetch_dataframe(conn, "SELECT COUNT(*) candidates, AVG(year_continuity_score) cont, AVG(gap_count) gaps, AVG(density_balance_score) balance FROM m0_graph.candidate_graph_features")
    print(f"number of tickers: {int(df.iloc[0]['tickers']) if not df.empty else 0}")
    print(f"number of ticker-year rows: {int(df.iloc[0]['rows']) if not df.empty else 0}")
    print(f"number of candidate windows processed: {int(feats.iloc[0]['candidates']) if not feats.empty else 0}")
    print(f"avg year_continuity_score: {float(feats.iloc[0]['cont'] or 0):.4f}")
    print(f"avg gap_count: {float(feats.iloc[0]['gaps'] or 0):.4f}")
    print(f"avg density_balance_score: {float(feats.iloc[0]['balance'] or 0):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_db_args(parser)
    parser.add_argument("--method", default="vector_probe_agent_v1")
    parser.add_argument("--m0-input-setting", default="query_with_corpus_metadata")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--build-version", default=DEFAULT_BUILD_VERSION)
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    with connect_db(args) as conn:
        apply_schema(conn)
        if args.clear_existing:
            clear_existing(conn, args.build_version)
        build_ticker_year_evidence(conn, args.build_version)
        inputs = selected_inputs(conn, args.method, args.m0_input_setting, args.limit)
        build_candidate_features(conn, inputs, args.build_version)
        export_outputs(conn)
        generate_report(conn, Path("m0-graph-lite/outputs/markdown/m0_graph_lite_report.md"))
        print_summary(conn)


if __name__ == "__main__":
    main()
