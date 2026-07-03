# M0 Graph-Lite

Graph-lite is a PostgreSQL-based temporal structure feature layer for M0.

It models:

- Ticker -> Year -> Evidence
- CandidateWindow -> covers Year

It does not use Neo4j, LLM relation extraction, QA gold labels, or final answer generation.

## Why It Exists

M0 scoring experiments showed:

- E2 improves over baseline.
- E3 reduces over-extension but can increase under-extension.
- Candidate windows already contain acceptable answers.
- Scoring lacks temporal structure signals.

Graph-lite adds content-side timeline structure features so M0 can reason about continuity, gaps, density balance, and candidate position.

## What It Computes

- `m0_graph.ticker_year_evidence`
- `m0_graph.candidate_year_coverage`
- `m0_graph.candidate_graph_features`

Features include:

- normalized candidate position
- year continuity
- gap count and max gap length
- prefix/suffix support
- coverage density
- density balance

## Data Leakage Policy

Gold labels are not used in graph-lite feature construction.

Gold labels are only used later in `m0-exp` evaluation.

## Run

```bash
python m0-graph-lite/build_graph_lite_features.py   --host localhost   --port 5432   --db bridge_temporal   --user bridge   --password bridge   --method vector_probe_agent_v1  --m0-input-setting query_with_corpus_metadata   --limit 100   --build-version graph_lite_v1   --clear-existing
```

## Outputs

Database:

- `m0_graph.ticker_year_evidence`
- `m0_graph.candidate_year_coverage`
- `m0_graph.candidate_graph_features`
- `m0_graph.query_relevant_chunks`
- `m0_graph.query_year_evidence`
- `m0_graph.candidate_query_graph_features`

CSV:

- `m0-graph-lite/outputs/csv/ticker_year_evidence.csv`
- `m0-graph-lite/outputs/csv/candidate_year_coverage.csv`
- `m0-graph-lite/outputs/csv/candidate_graph_features.csv`
- `m0-graph-lite/outputs/csv/candidate_graph_feature_summary.csv`

Markdown:

- `m0-graph-lite/outputs/markdown/m0_graph_lite_report.md`
- `m0-graph-lite/outputs/markdown/query_relevant_graph_lite_report.md`

## Query-Relevant Graph-Lite V2

Graph-lite v1 answered: does a ticker-year have content evidence?

E4 showed that availability-only structure is too dense to be very discriminative. Query-relevant graph-lite v2 answers a stronger question: does a ticker-year have evidence relevant to this M0 query?

V2 computes:

- query-relevant chunks
- query-year evidence scores
- candidate-window query relevance features

The flow is:

```text
chunk relevance -> year relevance -> candidate-window relevance
```

Gold labels are not used in feature construction.

Run:

```bash
python m0-graph-lite/build_query_relevant_graph_features.py   --host localhost   --port 5432   --db bridge_temporal   --user bridge   --password bridge   --method vector_probe_agent_v1   --m0-input-setting query_with_corpus_metadata   --limit 100   --top-k-per-year 5   --build-version graph_lite_v2_query_relevant   --clear-existing
```

V2 outputs:

- `m0-graph-lite/outputs/csv/query_relevant_chunks.csv`
- `m0-graph-lite/outputs/csv/query_year_evidence.csv`
- `m0-graph-lite/outputs/csv/candidate_query_graph_features.csv`
- `m0-graph-lite/outputs/csv/candidate_query_graph_feature_summary.csv`

## Hybrid Query-Relevant Graph-Lite E5b

E5 lexical relevance may be too weak. E5b upgrades chunk relevance to hybrid relevance:

```text
lexical + dense semantic + signal bonus
-> hybrid chunk relevance
-> hybrid year relevance
-> hybrid candidate-window relevance
```

Dense embeddings are optional. If `sentence-transformers` is unavailable or model loading fails, the script falls back to `--dense-mode none` and continues.

Run:

```bash
python m0-graph-lite/build_hybrid_query_relevant_graph_features.py   --host localhost  --port 5432  --db bridge_temporal  --user bridge  --password bridge --method vector_probe_agent_v1  --m0-input-setting query_with_corpus_metadata  --limit 100  --top-k-per-year 5  --max-chunks-per-year 50  --build-version graph_lite_v2_hybrid  --dense-mode sentence_transformer  --embedding-model BAAI/bge-m3  --clear-existing
```

For a dependency-free run:

```bash
python m0-graph-lite/build_hybrid_query_relevant_graph_features.py  --host localhost  --port 5432  --db bridge_temporal  --user bridge   --password bridge  --dense-mode none  --clear-existing
```

Hybrid outputs:

- `m0-graph-lite/outputs/csv/hybrid_query_relevant_chunks.csv`
- `m0-graph-lite/outputs/csv/hybrid_query_year_evidence.csv`
- `m0-graph-lite/outputs/csv/hybrid_candidate_query_graph_features.csv`
- `m0-graph-lite/outputs/csv/hybrid_candidate_query_graph_feature_summary.csv`
- `m0-graph-lite/outputs/markdown/hybrid_query_relevant_graph_lite_report.md`

## Boundary-Aware E6 Features

E6 uses E5b hybrid year relevance as a time series. It remains rule-based and does not use gold labels.

It computes:

- year-level trend signals from top hybrid-relevant chunks
- inside relevance and inside coherence for candidate windows
- left/right boundary contrast against neighboring years
- trend coherence inside the window
- intent fit and length fit
- `boundary_score_raw`

Formula:

```text
boundary_score_raw =
0.15 * intent_fit
+ 0.20 * inside_relevance_mean
+ 0.15 * inside_coherence
+ 0.20 * boundary_contrast
+ 0.20 * trend_coherence
+ 0.10 * length_fit
```

Run:

```bash
python m0-graph-lite/build_boundary_aware_features.py \
  --host localhost \
  --port 5432 \
  --db bridge_temporal \
  --user bridge \
  --password bridge \
  --method vector_probe_agent_v1 \
  --m0-input-setting query_with_corpus_metadata \
  --limit 100 \
  --context-size 2 \
  --source-build-version graph_lite_v2_hybrid \
  --build-version boundary_v1 \
  --clear-existing
```

Boundary outputs:

- `m0-graph-lite/outputs/csv/year_trend_signals.csv`
- `m0-graph-lite/outputs/csv/candidate_boundary_features.csv`
- `m0-graph-lite/outputs/csv/candidate_boundary_feature_summary.csv`
- `m0-graph-lite/outputs/markdown/boundary_aware_feature_report.md`
