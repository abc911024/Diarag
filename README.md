# DiaRAG

Structured Diachronic Reasoning for Retrieval-Augmented Generation.

This repo implements the pipeline from the paper *"DiaRAG: Structured Diachronic Reasoning for
Retrieval-Augmented Generation"*. Standard RAG scores passages by semantic similarity only, which
breaks down for **diachronic queries** — questions about trends, evolution, or change over a time
interval (e.g. "how has this company's carbon emission trajectory evolved since its 2018 net-zero
pledge?"). DiaRAG closes two gaps left open by prior diachronic-RAG work: inferring the relevant
time range when it isn't explicit in the query, and reasoning across multiple time periods instead
of leaving that to the generator.

## Pipeline

Four modules, each producing the input to the next:

| Module | Name | What it does | Key scripts |
|---|---|---|---|
| **M0** | Time Range Inference | Classifies a query's temporal type (explicit / relative / event-anchored / latent) and infers a `[start_year, end_year]` window, with iterative coverage-based verification against the corpus. | `scripts/run_m0_agent.py`, `scripts/run_m0_agent_batch.py`, `scripts/m0_agent_tools.py`, `scripts/run_m0_experiment_gptoss.py` |
| **M1** | Temporal Retrieval | Embeds the query, retrieves top-k evidence from the corpus filtered to M0's predicted year range, and scores Temporal Coverage@k / Precision@k. | `scripts/run_m1_qwen_experiment.py`, `build_qwen_embeddings.py` |
| **M2** | Temporal Evidence Structuring | Converts M1's flat top-k evidence into a year-by-year temporal evidence graph (per-period summaries) via an LLM. Does not retrieve new evidence. | `scripts/run_m2_gptoss_structure.py` |
| **M3** | Structured Temporal Reasoning | Decomposes trend analysis into pairwise change analysis, turning-point detection, and trend synthesis (chain-of-thought), then selects the final MCQA answer. | `scripts/run_m3_gptoss_mcqa.py` (ablation without M2: `run_m3_gptoss_wo_m2.py`) |

Each stage hands off via CSV: `m0_runs.csv → m1_qwen_results.csv → m2_structured_evidence.csv → m3_answers.csv`.

`m0-graph-lite/` is a supporting feature layer for M0: it builds a lightweight Postgres-based
Ticker→Year→Evidence structure (continuity, gap, coverage-density, boundary-contrast features) that
M0's scoring methods can use, without ever touching gold labels. See
`m0-graph-lite/README_M0_GRAPH_LITE.md` for the E2/E3/E5b/E6/E7 feature variants.

`Dataest-Bridge/` and `exp_bridge_v3_m0/` build and evaluate **ADQAB-Implicit**: a controlled
rewrite of the ADQAB benchmark where each question is rewritten into relative / event-anchored /
latent form while preserving the original gold answer and time range, used to evaluate M0 on
implicit temporal queries.

`exp_ect_qa/` is a separate feasibility experiment validating whether the external
`austinmyc/ECT-QA` dataset can be adapted into this framework without forcing its schema; see
`exp_ect_qa/README.md`.

## Repo layout

```
.
├── scripts/                 # M0-M3 pipeline drivers, DB seed/export utilities, smoke tests
├── m0-graph-lite/            # M0 feature layer (Postgres-based temporal structure) + sql/
├── Dataest-Bridge/            # ADQAB-Implicit rewrite pipeline (bridge_pipeline_v3.1.py)
├── exp_bridge_v3_m0/          # M0-only evaluation harness on the Bridge v3 rewrite dataset
├── exp_ect_qa/                # ECT-QA feasibility experiment (9-step script pipeline)
├── postgres/init/             # Postgres schema (qa, content, runtime, m0_graph)
├── docker-compose.yml         # postgres+pgvector and adminer for local dev
├── build_qwen_embeddings.py   # builds corpus_embeddings.npy for M1
├── requirements.txt           # full environment freeze
└── pre_A7_m0.md                # internal M0 experiment report (E2/E3/E6/E7 results)
```

`dataset/` and `outputs/` are **not** in this repo (see below).

## Data

Raw corpus and experiment-output files are intentionally excluded from git (`.gitignore`:
`dataset/`, `outputs/`, `exp_ect_qa/processed/`, `*.jsonl`, `*.csv`) — the raw corpus alone is
~370MB and none of it is needed to read or review the code. This was also the fix for the original
"can't push" problem: an earlier `.gitignore` excluded a folder named `data/` that didn't exist,
while the real `dataset/` folder was never ignored.

To run anything locally, copy these back from the team Drive (`實作/ds_final/`) into the repo root:

- `dataset/corpus_with_time.jsonl`, `dataset/corpus_with_time.csv`, `dataset/C_final.jsonl`,
  `dataset/dqabench_MCQA.json` — corpus + MCQA benchmark used by M1/M3
- `Dataest-Bridge/dqabench_MCQA.json` — same benchmark, used by the Bridge rewrite pipeline
- `exp_ect_qa/processed/*` — ECT-QA import intermediates (only needed to re-run `exp_ect_qa/scripts/`)
- any `outputs/` folder under `scripts/`, `m0-graph-lite/`, `exp_bridge_v3_m0/`, `exp_ect_qa/` if you
  want to inspect past experiment runs instead of regenerating them

There is no script in this repo that generates `corpus_with_time.jsonl` from scratch — it is
prepared data, not a public dataset with a download URL, so it has to come from the team's copy.

## Setup

```bash
cp .env.example .env        # fill in real Postgres credentials if not using the defaults below
docker compose up -d        # starts postgres-pgvector (port 5432) + adminer (port 8080)
pip install -r requirements.txt
python scripts/seed_content_postgres.py   # load corpus into Postgres
python scripts/seed_qa_postgres.py        # load QA/benchmark data into Postgres
python scripts/smoke_test_runtime.py      # sanity check the DB is reachable and seeded
```

`docker-compose.yml` mounts `postgres/init/` so schemas (`qa`, `content`, `runtime`, `m0_graph`) are
created automatically on first startup.

## Running the pipeline

```bash
# M0: infer time ranges
python scripts/run_m0_agent_batch.py ...

# M1: build embeddings once, then retrieve
python build_qwen_embeddings.py --corpus dataset/corpus_with_time.jsonl --out-index corpus_index_qwen.csv --out-embeddings corpus_embeddings_qwen.npy
python scripts/run_m1_qwen_experiment.py --m0-results <m0_runs.csv> --corpus-index corpus_index_qwen.csv --corpus-embeddings corpus_embeddings_qwen.npy

# M2: structure the retrieved evidence (requires an Ollama/gpt-oss backend)
python scripts/run_m2_gptoss_structure.py --m1-results m1_qwen_results.csv --backend ollama --model gpt-oss:20b

# M3: final MCQA answer selection
python scripts/run_m3_gptoss_mcqa.py --m2-results m2_structured_evidence_with_source.csv --mcqa-json dataset/dqabench_MCQA.json --backend ollama --model gpt-oss:20b
```

Each script's `--help` / module docstring documents its exact CLI flags.

## Status

- **M0** is the most mature module — several scoring strategies have been tried (E2 stable
  temporal-prior scoring through E7 boundary-aware fusion); see `pre_A7_m0.md` for full results.
  E2 is currently the strongest, at 0.19 acceptable accuracy against an oracle upper bound of 1.0 —
  time-range selection (not candidate generation) is the open problem.
- **M1–M3** scripts exist and chain via CSV, but a full, verified end-to-end M0→M3 run has not yet
  been completed and reported.
- No vector index (FAISS or similar) exists yet; M1 currently does brute-force cosine similarity
  over `corpus_embeddings_qwen.npy`.
