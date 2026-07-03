#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from db_config import add_connection_args, get_connection
from run_m0_agent import run_agent

FORBIDDEN = ["gold_range", "acceptable_ranges", "gold_start_date", "gold_end_date", "source_temporal_label", "temporal_scope_type"]


def scalar(cur, sql: str, params=None):
    cur.execute(sql, params or ())
    return cur.fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_connection_args(parser)
    args = parser.parse_args()
    with get_connection(args) as conn:
        with conn.cursor() as cur:
            exists = scalar(cur, "SELECT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'runtime')")
            assert exists, "runtime schema does not exist; run apply_runtime_schema.py"
            for table in ("qa_content_coverage", "m0_candidate_windows", "m0_evidence_probes", "m0_agent_runs", "m0_agent_metrics"):
                ok = scalar(cur, "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='runtime' AND table_name=%s)", (table,))
                assert ok, f"missing runtime table {table}"
            assert scalar(cur, "SELECT COUNT(*) FROM qa.m0_inputs") > 0, "qa.m0_inputs is empty"
            assert scalar(cur, "SELECT COUNT(*) FROM content.evidence_chunks") > 0, "content.evidence_chunks is empty"
            cov_count = scalar(cur, "SELECT COUNT(*) FROM runtime.qa_content_coverage")
        if cov_count == 0:
            script = Path(__file__).resolve().parent / "build_qa_content_coverage.py"
            subprocess.run([
                sys.executable, str(script), "--host", args.host or "localhost", "--port", str(args.port or 5432),
                "--db", args.db or "bridge_temporal", "--user", args.user or "bridge",
                "--password", args.password or "bridge", "--limit", "5",
            ], check=True)
        with conn.cursor() as cur:
            cur.execute("SELECT m0_input_id FROM qa.m0_inputs ORDER BY m0_input_id LIMIT 1")
            m0_input_id = cur.fetchone()[0]
        run_agent(conn, m0_input_id, "vector_probe_agent_v1", 5, True)
        with conn.cursor() as cur:
            assert scalar(cur, "SELECT COUNT(*) FROM runtime.m0_agent_runs") > 0, "no agent runs"
            assert scalar(cur, "SELECT COUNT(*) FROM runtime.m0_agent_metrics") > 0, "no agent metrics"
            cur.execute("SELECT * FROM runtime.v_m0_agent_results LIMIT 1")
            assert cur.fetchone() is not None, "runtime.v_m0_agent_results empty"
            for key in FORBIDDEN:
                assert scalar(cur, "SELECT COUNT(*) FROM qa.m0_inputs WHERE m0_input_json ? %s", (key,)) == 0
            assert scalar(cur, "SELECT COUNT(*) FROM runtime.m0_agent_runs WHERE tool_trace IS NOT NULL") > 0
    print("Runtime smoke test passed")


if __name__ == "__main__":
    main()
