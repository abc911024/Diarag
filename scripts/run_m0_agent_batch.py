#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

from psycopg.rows import dict_row

from db_config import add_connection_args, get_connection
from run_m0_agent import run_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-input-setting", default="query_with_corpus_metadata")
    parser.add_argument("--method", default="vector_probe_agent_v2", choices=["metadata_only_agent_v1", "vector_probe_agent_v1", "metadata_only_agent_v2", "vector_probe_agent_v2"])
    parser.add_argument("--agent-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--candidate-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--scoring-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--clear-existing", action="store_true")
    parser.add_argument("--store-oracle-analysis", action=argparse.BooleanOptionalAction, default=True)
    add_connection_args(parser)
    args = parser.parse_args()

    with get_connection(args) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            if args.clear_existing:
                method_pattern = f"%__{args.method}__%"
                cur.execute(
                    """
                    DELETE FROM runtime.m0_evidence_probes
                    WHERE candidate_window_id IN (
                        SELECT candidate_window_id
                        FROM runtime.m0_candidate_windows
                        WHERE candidate_window_id LIKE %s
                    )
                    """,
                    (method_pattern,),
                )
                cur.execute("DELETE FROM runtime.m0_candidate_windows WHERE candidate_window_id LIKE %s", (method_pattern,))
                cur.execute("DELETE FROM runtime.m0_candidate_oracle_analysis WHERE method = %s", (args.method,))
                cur.execute("DELETE FROM runtime.m0_agent_metrics WHERE agent_run_id IN (SELECT agent_run_id FROM runtime.m0_agent_runs WHERE method = %s)", (args.method,))
                cur.execute("DELETE FROM runtime.m0_agent_runs WHERE method = %s", (args.method,))
                conn.commit()
            cur.execute(
                """
                SELECT m0_input_id
                FROM qa.m0_inputs
                WHERE m0_input_setting = %s
                ORDER BY m0_input_id
                LIMIT %s
                """,
                (args.m0_input_setting, args.limit),
            )
            ids = [r["m0_input_id"] for r in cur.fetchall()]

        results = []
        for m0_input_id in ids:
            try:
                results.append(run_agent(
                    conn, m0_input_id, args.method, args.top_k, args.allow_fallback,
                    args.agent_version, args.candidate_version, args.scoring_version,
                    args.store_oracle_analysis,
                ))
            except Exception as exc:
                print(f"failed {m0_input_id}: {exc}")
                results.append({"decision_status": "failed", "metrics": {}, "coverage_status": "failed"})

    predicted = sum(1 for r in results if r.get("decision_status") == "predicted")
    failed = sum(1 for r in results if r.get("decision_status") == "failed")
    overlaps = [r.get("metrics", {}).get("overlap_score") for r in results if r.get("metrics", {}).get("overlap_score") is not None]
    boundary = [r.get("metrics", {}).get("mean_boundary_error") for r in results if r.get("metrics", {}).get("mean_boundary_error") is not None]
    acceptable = [r.get("metrics", {}).get("acceptable_accuracy") for r in results if "acceptable_accuracy" in r.get("metrics", {})]
    coverage = Counter(r.get("coverage_status", "unknown") for r in results)
    print(f"total runs: {len(results)}")
    print(f"predicted: {predicted}")
    print(f"failed: {failed}")
    print(f"avg overlap_score: {(sum(overlaps) / len(overlaps)) if overlaps else 0:.4f}")
    print(f"acceptable_accuracy rate: {(sum(1 for x in acceptable if x) / len(acceptable)) if acceptable else 0:.4f}")
    print(f"avg boundary error: {(sum(boundary) / len(boundary)) if boundary else 0:.4f}")
    print(f"coverage status distribution: {dict(coverage)}")


if __name__ == "__main__":
    main()
