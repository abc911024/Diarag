# pre_A7: Internal Technical Foundation Report

This document is not the final paper. It is an internal technical report intended to preserve the factual state of the repository before a future academic paper ("A7") is written. The report is grounded in the repository contents, experiment scripts, generated CSV/Markdown outputs, prompts, and observed run artifacts available in this workspace.

The central project should be understood as:

```text
Implicit Temporal Scope Inference for Diachronic Question Answering
```

The main research question is:

> How can a system infer an analysis-worthy temporal scope for an implicit diachronic query?

This report deliberately does not redefine the project as Multi-Agent RAG, Temporal Knowledge Graph QA, forecasting, or general GraphRAG. Those topics appear only where implemented as supporting mechanisms or where explicitly marked as future design ideas.

## 1. Executive Summary

The implemented project focuses on M0: temporal scope inference / temporal window scoring. Given an implicit diachronic query, the system selects a predicted temporal scope, represented in the current Bridge-style implementation as `[start_year, end_year]`. The upstream M0 agent generates candidate temporal windows. The core experimental question is whether different scoring policies can select a better candidate window without using gold labels at selection time.

The strongest completed intrinsic finding is that candidate generation is not the main bottleneck for the evaluated 100-query M0 experiment. The diagnostic module reports that acceptable candidate windows are present at a high rate, while the deployed baseline frequently selects the wrong one. In `m0-diagnostic/outputs/markdown/m0_diagnostic_report.md`, the candidate oracle summary shows an acceptable candidate rate of 1.0 for both rewrite type A and rewrite type B under `query_with_corpus_metadata`, while selected acceptable accuracy is only 0.08 for each. The oracle upper bound in `m0-exp/outputs/csv/experiment_summary.csv` reaches acceptable accuracy 1.0, while deployable methods remain far lower.

The strongest non-oracle exploratory result before the final consolidation is E2 (`scoring_v2`), with overall acceptable accuracy 0.19 across 100 examples. E2 uses stable temporal-prior features: intent fit, length fit, evidence year coverage, and window type prior. E2 improves over E0's 0.08 acceptable accuracy but remains far from the oracle upper bound.

Later experiments show that simply adding more complex features does not reliably improve temporal boundary selection. E3 reduces the rolling-window selection rate but drops acceptable accuracy from 0.19 to 0.16. E4 graph-lite features, E5 query relevance, and E5b hybrid query relevance do not surpass E2. E6 boundary-aware scoring is recorded by the project context as 0.0000 acceptable accuracy and should not be overinterpreted because the report artifacts emphasize E6b rather than a stable E6 result. E6b introduces boundary contrast, trend coherence, candidate length ratio, long-window penalty, and directional adjustment. It improves over E5b in the exploratory comparison but remains below E2. E7 uses chunk-level evidence qualification / feature-specific routing before recomputing E6b-style boundary features; it improves over E6b but still does not surpass E2 in exploratory outputs.

The final consolidated M0 folder, `m0_final/`, compares E2, E6b, E7, and FINAL. The final summary in `m0_final/outputs/csv/final_experiment_summary.csv` reports:

| Experiment | Acceptable Accuracy | Avg Overlap | Avg Boundary Error | Avg Over Extension | Avg Under Extension | Avg Predicted Length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2_stable_temporal_prior | 0.19 | 0.5686 | 2.065 | 2.81 | 1.32 | 7.73 |
| E6b_raw_boundary_scoring | 0.09 | 0.5422 | 2.31 | 3.36 | 1.26 | 8.34 |
| E7_qualified_boundary_scoring | 0.14 | 0.5192 | 2.495 | 3.41 | 1.58 | 8.07 |
| FINAL_prior_guided_qualified_boundary | 0.17 | 0.5600 | 2.115 | 2.89 | 1.34 | 7.79 |

The final fused method improves over E6b and E7 in the consolidated run, but it does not surpass E2. Therefore, the strongest supported conclusion today is not that FINAL is the best method. The supported conclusion is narrower: stable temporal priors are still difficult to beat; qualified evidence helps boundary-aware scoring; and fusing prior and qualified evidence partially recovers performance but needs further calibration.

Downstream M0-M3 validation is partially implemented in `exp_original_m0_m3/`. It prepares scope runs, runs scope-constrained BM25 retrieval, builds deterministic time-ordered contexts, generates answers with Qwen 7B, judges answers with Qwen 14B, and adds a supplementary RAGAS script. However, the current output state shows that the full downstream evaluation has not yet been completed. `scope_runs.csv`, `retrieved_evidence.csv`, and `timeline_contexts.csv` exist for 300 rows (100 queries x 3 methods). The answer/judgment/RAGAS outputs currently contain only a one-row smoke test, with the generated answer empty due to a short generation run. Thus, no supported downstream conclusion about answer quality can yet be made.

ECT-QA external validation is implemented as a feasibility/adaptation experiment in `exp_ect_qa/`. It establishes that ECT-QA has query text, time-indexed evidence units, company metadata, and reference answers, but no reliable gold temporal scope derivable without invention. Therefore ECT-QA is currently usable for downstream answer evaluation or qualitative framework validation, not M0 acceptable-accuracy evaluation.

The current limitations are substantial. M0 acceptable accuracy remains low. Boundary-aware methods tend to over-extend windows. Query relevance features can collapse toward overly short or semantically broad windows. Downstream answer validation is not yet complete. Graph-assisted temporal reasoning, event/phase representation, and temporal evidence graphs remain design ideas, not demonstrated contributions.

## 2. Problem Definition

### 2.1 Diachronic QA

Diachronic question answering asks about change, development, trend, or behavior across time. It differs from fact retrieval because the answer is not merely a single fact located in one document. A diachronic answer may require deciding which time period matters, gathering evidence from multiple time points, comparing states across that period, and summarizing a temporal trajectory.

In the current dataset, many questions ask about stock price movements. Examples from `data/qa/raw_qa_items.jsonl` and `data/qa/qa_quality_report.csv` include:

- "What was the general stock price movement for AGG during 2021?"
- "How did AGG's stock price generally trend?"
- "How did AGG's stock price generally trend in earlier periods?"
- "How did the price of AGG trend after April 2019?"
- "What was the predominant price trend for AGG over a multi-year period?"

The domain is financial in the original dataset, but the framework vocabulary was later generalized in `m0_final/README.md`: `ticker` becomes one dataset-specific realization of `target_subject`, `year` becomes a `time_unit`, and text chunks become `evidence_unit`s. The research problem is broader than finance: infer the time span over which a target subject's temporal development should be analyzed.

Standard retrieval can return semantically similar documents without deciding whether the evidence covers the correct temporal scope. A retriever may find strong evidence for some years but omit earlier or later years needed to answer a trend question. Conversely, it may retrieve all available years and drown the answer generator in temporally irrelevant evidence. This motivates M0: temporal scope inference before downstream retrieval and generation.

### 2.2 Implicit Time Range Problem

Explicit ranges are relatively easy because the query names a time span. For example, "during 2021" or "from 2012 to 2019" provides direct constraints. A system can extract the range and filter evidence.

Implicit ranges are difficult because the query asks about a temporal development without giving the complete range. The system must infer an "analysis-worthy" temporal scope. In the current project, the implicit diachronic query is usually a rewritten query, such as "How did AGG's stock price generally trend?" The gold scope may be `[2012, 2019]`, `[2015, 2022]`, `[2019, 2022]`, or another range depending on the source item's temporal context. The rewritten query deliberately removes explicit time labels in many cases, forcing M0 to infer scope from query intent, candidate windows, corpus metadata, and evidence signals.

The repository's observable query taxonomy comes from `data/qa/raw_qa_items.jsonl` and the processed bridge records:

| Source query type | Count in raw QA | Notes |
| --- | ---: | --- |
| Specific Time Period | 75 | Usually explicit single-year questions in the source dataset. |
| Before | 150 | Questions anchored before a year/month. |
| After | 150 | Questions anchored after a year/month. |
| Time Interval | 150 | Questions over explicit intervals in the source dataset. |

The `query_type_id` distribution is:

| query_type_id | Count |
| --- | ---: |
| SP_Year | 75 |
| B_YearAnchor | 75 |
| A_YearAnchor | 75 |
| B_MonthAnchor | 75 |
| A_MonthAnchor | 75 |
| TI_Years | 75 |
| TI_Months | 75 |

The rewritten M0 dataset has two rewrite types per bridge record:

- Type A: generally phrased trend questions, often broad or underspecified.
- Type B: alternate rewrites that may add vague directional language such as "earlier periods", "later periods", or "over a multi-year period".

The implemented intent detector in `m0-exp/scoring_policies.py` and `m0_final/scoring.py` recognizes relative intent categories:

- `earlier`
- `recent`
- `later`
- `broad`
- `broad_unknown`
- `unknown`

These are rule-based categories, not learned labels. They are derived from surface phrases such as "earlier periods", "before", "recent years", "later periods", "after", "over time", "generally trend", and "multi-year".

The M0 task is:

```text
Input:
  rewritten_query
  target_subject / ticker
  candidate temporal scopes
  optional corpus metadata and evidence probes

Output:
  predicted_temporal_scope = [t_start, t_end]
```

Gold labels are used only for evaluation after selection.

## 3. Dataset Construction

### 3.1 Source Datasets and Local Data

The original dataset is stored under `data/qa/` and `data/content/`. The repository contains:

- `data/qa/raw_qa_items.jsonl`: 525 raw QA items.
- `data/qa/qa_quality_report.csv`: 706 rewritten bridge records.
- `data/qa/bridge_m0_inputs_qa.jsonl`: 2,118 M0 input variants.
- `data/qa/bridge_m0_eval_targets_qa.jsonl`: 706 temporal evaluation targets.
- `data/runtime_exports/qa_content_coverage.csv`: coverage diagnostics for 706 M0 inputs.
- `data/content/evidence_chunks.jsonl`: 209,427 evidence chunks.

The raw QA items cover 25 tickers with 21 items each. The raw query type distribution is shown in Section 2.2. The bridge records expand raw items into A/B rewritten forms, yielding 706 rewritten records. The quality report reports 692 `pass` and 14 `fail` rewrite quality statuses.

The M0 input table contains three input settings per bridge record:

| M0 input setting | Count |
| --- | ---: |
| query_only | 706 |
| query_with_corpus_metadata | 706 |
| query_with_verification | 706 |

All M0 inputs have `temporal_need_type = historical_trend` and `evidence_policy = broad_historical_coverage` in the observed `bridge_m0_inputs_qa.jsonl` statistics.

### 3.2 Filtering Rules and Coverage

The current M0 experiments use the `query_with_corpus_metadata` setting and, in the main 100-query experiments, a selected subset of 100 inputs. The coverage file `data/runtime_exports/qa_content_coverage.csv` contains 706 rows and reports `full_gold_coverage` for all 706 rows. It also marks `can_run_m1_retrieval = True` for all 706 rows. This means the local content corpus contains evidence for the gold years at the level checked by the coverage diagnostic. It does not mean the retriever will necessarily select the right evidence.

The `m0_final` comparison uses 100 selected M0 inputs. In `exp_original_m0_m3`, the 100-query downstream scope runs are expanded to 300 rows by comparing `NoScope`, `E2`, and `FINAL`.

### 3.3 Rewrite Process

The rewrite process creates implicit diachronic queries from source questions. `data/qa/qa_quality_report.csv` contains:

- `bridge_id`
- `source_id`
- `rewrite_type`
- `rewrite_status`
- `rewrite_quality_status`
- `rewrite_quality_flags`
- `rewritten_query`

All 706 rewrites have `rewrite_status = generated`; 692 pass the recorded rewrite quality check. The quality flags include checks such as whether the rewrite contains an explicit year, preserves the entity, keeps diachronic intent, and is a non-empty question. The project therefore constructs a test condition where explicit temporal labels are removed or softened, creating implicit temporal scope inference examples.

Example observed records:

| bridge_id | rewrite_type | rewritten_query |
| --- | --- | --- |
| DQABench_00004_A | A | How did AGG's stock price generally trend? |
| DQABench_00004_B | B | How did AGG's stock price generally trend in earlier periods? |
| DQABench_00005_B | B | How did AGG's stock price generally trend over a multi-year period? |
| DQABench_00008_A | A | How did the stock price of AGG evolve in the period? |

### 3.4 Annotation Process

Temporal targets are stored in `data/qa/bridge_m0_eval_targets_qa.jsonl`. Each record includes:

- `bridge_id`
- `source_id`
- `gold_start_date`
- `gold_end_date`
- `gold_start_year`
- `gold_end_year`
- `gold_range`
- `acceptable_ranges`
- `temporal_scope_type`
- `source_temporal_label`
- `anchor_year`
- `anchor_direction`
- `annotation_source`

The target distribution is:

| temporal_scope_type | Count |
| --- | ---: |
| after_anchor | 284 |
| before_anchor | 250 |
| explicit_range | 172 |

The annotation source is represented in examples as `plot_time_bounds`. This suggests that the gold temporal ranges are derived from the source dataset's plotted time bounds rather than manually inferred by the current M0 model. Acceptable ranges include small boundary variants. For example:

```json
{
  "bridge_id": "DQABench_00004_A",
  "gold_range": [2012, 2019],
  "acceptable_ranges": [[2012, 2019], [2011, 2019], [2012, 2020], [2011, 2020]],
  "temporal_scope_type": "before_anchor",
  "source_temporal_label": "before 2020"
}
```

### 3.5 Example: Original Query, Rewrite, Acceptable Range, Rationale

From `data/qa/raw_qa_items.jsonl`, `DQABench_00004` has ticker `AGG` and a source temporal label linked to before 2020. The rewritten bridge examples include:

| Field | Value |
| --- | --- |
| source_id | DQABench_00004 |
| ticker / target_subject | AGG |
| rewritten query A | How did AGG's stock price generally trend? |
| rewritten query B | How did AGG's stock price generally trend in earlier periods? |
| gold range | [2012, 2019] |
| acceptable ranges | [2012, 2019], [2011, 2019], [2012, 2020], [2011, 2020] |

The rationale is that the source item has a before-anchor temporal context ("before 2020"), but the rewrite removes the explicit anchor. M0 must infer that the analysis-worthy period is the historical period ending around 2019 rather than simply selecting a recent rolling window or full entity span.

### 3.6 Assumptions

The current dataset construction assumes:

1. Year-level temporal scope is sufficient for M0 evaluation.
2. Source plot time bounds can define gold temporal scope for rewritten implicit queries.
3. A small set of acceptable boundary variants accounts for annotation tolerance.
4. Ticker/entity identity is a valid analysis focus for the original dataset.
5. Rewritten queries preserve diachronic intent even when explicit temporal labels are removed.
6. The `query_with_corpus_metadata` input setting may include entity-level coverage metadata but not item-level gold labels.

These assumptions are reasonable for the present benchmark but should not be generalized without validation on non-financial datasets.

## 4. System Evolution

This section reconstructs the experiment sequence from `m0-diagnostic/`, `m0-exp/`, `m0-graph-lite/`, and `m0_final/`.

### 4.1 E0: Runtime Baseline

#### Goal

Establish the performance of the original deployed M0 agent selection.

#### Method

E0 reads the existing runtime selection from `runtime.m0_agent_runs` and candidate windows generated by the upstream M0 agent. It is represented in `m0-exp/outputs/csv/experiment_summary.csv` as `E0_v1_baseline`.

#### Result

E0 has acceptable accuracy 0.08 for rewrite type A and 0.08 for rewrite type B. Combined, this is 0.08. Average overlap is approximately 0.487 for A and 0.543 for B. Average predicted length is 8.0 for A and 7.84 for B, while average gold length is 6.24.

#### Interpretation

The runtime baseline often chooses a plausible but incorrect temporal window. Diagnostic outputs show candidate generation is not the bottleneck: acceptable candidates exist, but selection is weak. This motivates scoring experiments over the same candidate set.

### 4.2 E1: Oracle Upper Bound

#### Goal

Measure the best possible performance if the system could select the best candidate using gold labels after the fact.

#### Method

E1 is a non-deployable oracle. It uses gold labels only for post-hoc candidate selection. It is not a valid runtime method.

#### Result

E1 reaches acceptable accuracy 1.0 for both rewrite types. Average overlap is approximately 0.979 for both A and B. Exact match is 0.92. The candidate oracle summary in `m0-diagnostic` also reports acceptable candidate rate 1.0.

#### Interpretation

The candidate set usually contains an acceptable answer. The main bottleneck is scoring/selection, not candidate generation. This is one of the strongest project findings.

### 4.3 E2: Stable Temporal Prior Baseline

#### Goal

Test whether simple stable temporal-prior features can improve selection without gold labels.

#### Method

E2 uses `score_candidate_v2` in `m0-exp/scoring_policies.py`, reproduced in `m0_final/scoring.py` as `score_e2_prior`. The formula is:

```text
score =
  0.40 * intent_fit
+ 0.25 * length_fit
+ 0.20 * evidence_year_coverage
+ 0.15 * window_type_prior
```

The features are:

- `intent_fit`
- `length_fit`
- `evidence_year_coverage`
- `window_type_prior`

#### Result

In `m0-exp`, E2 has 0.14 acceptable accuracy for A and 0.24 for B, combined 0.19. In `m0_final`, E2 remains 0.19 with average overlap 0.5686, average boundary error 2.065, over-extension 2.81, and under-extension 1.32.

#### Interpretation

E2 is the strongest deployable baseline in the exploratory experiment and remains the strongest method in the final consolidated intrinsic comparison. This supports the claim that stable temporal priors matter for implicit temporal scope inference. However, `e2_vs_e3_summary.csv` shows E2 selects `rolling_8y` at a high rate (0.91), so E2 may rely strongly on a stable window-type prior rather than true boundary understanding.

### 4.4 E3: Temporal Position Awareness

#### Goal

Reduce E2's rolling-window bias by adding stronger temporal position and directional cues.

#### Method

E3 uses `score_candidate_v3`. It adds:

- `intent_direction_fit`
- `position_fit`
- `length_fit_v3`
- `window_type_regularization`

The formula shown in `M0_SCORING_EXPERIMENT_ANALYSIS.md` and implemented in `m0-exp/scoring_policies.py` is:

```text
score =
  0.35 * intent_direction_fit
+ 0.25 * position_fit
+ 0.20 * length_fit_v3
+ 0.15 * evidence_year_coverage
+ 0.05 * window_type_regularization
```

#### Result

E3 combined acceptable accuracy is 0.16, down from E2's 0.19. The rolling_8y selection rate drops from 0.91 to 0.68, and prefix/suffix selections increase.

#### Interpretation

E3 succeeds at reducing the rolling_8y bias but does not improve overall accuracy. This suggests that reducing prior bias alone is not enough; the alternative directional/position signals are not sufficiently reliable.

### 4.5 E4: Graph-Lite / Continuity Features

#### Goal

Test whether temporal continuity, gaps, prefix/suffix support, and density balance can improve boundary selection.

#### Method

E4 uses graph-lite features built in `m0-graph-lite/`, including `build_graph_lite_features.py` and graph feature outputs. The scoring formula described in `m0-exp/outputs/markdown/m0_exp_report.md` is:

```text
score =
  0.25 * intent_fit
+ 0.25 * graph_position_fit
+ 0.20 * prefix_suffix_support
+ 0.15 * density_balance_score
+ 0.10 * length_fit
+ 0.05 * continuity_gap_penalty
```

#### Result

E4 has 0.10 acceptable accuracy for A and 0.14 for B, combined 0.12. It does not improve over E2 or E3.

#### Interpretation

The graph-lite features are reasonable diagnostics but do not solve boundary correctness in this benchmark. The report notes that continuity was often saturated, which limits its discriminative power.

### 4.6 E5: Query-Relevant Graph Scoring

#### Goal

Address the limitation that raw content availability is not necessarily query-relevant. E5 tests whether query-relevant chunks aggregated to years and windows improve temporal scope selection.

#### Method

E5 uses query-relevant graph features built by `m0-graph-lite/build_query_relevant_graph_features.py`. It includes:

- `directional_relevance_fit`
- `query_relevant_year_ratio`
- `window_relevance_balance`
- `semantic_gap_penalty`

The formula is:

```text
score =
  0.20 * intent_fit
+ 0.25 * directional_relevance_fit
+ 0.20 * query_relevant_year_ratio
+ 0.15 * window_relevance_balance
+ 0.10 * length_fit
+ 0.10 * semantic_gap_penalty
```

#### Result

E5 has 0.04 acceptable accuracy for A and 0.12 for B, combined 0.08. This is worse than E2 and roughly at the E0 level.

#### Interpretation

Query relevance by itself does not solve temporal boundary selection. The outputs show high query-relevant year ratios and low semantic gap counts in many selected windows, indicating that the relevance signal may be too dense or too broad to distinguish the correct temporal scope.

### 4.7 E5b: Hybrid Lexical + Dense Query Relevance

#### Goal

Improve E5 by upgrading query relevance from lexical-only to hybrid lexical/dense relevance with additional signal bonuses.

#### Method

E5b uses `m0-graph-lite/build_hybrid_query_relevant_graph_features.py` and `hybrid_relevance_utils.py`. It computes hybrid candidate query graph features, including:

- `hybrid_directional_relevance_fit`
- `hybrid_relevant_year_ratio`
- `hybrid_window_relevance_balance`
- `hybrid_semantic_gap_penalty`

#### Result

E5b has 0.06 acceptable accuracy for A and 0.12 for B, combined 0.09. It is a small improvement over E5 but still far below E2.

#### Interpretation

Hybrid query relevance helps slightly over E5 but remains insufficient. The method tends to select shorter windows, producing more under-extension than E2. Query relevance can identify semantically related evidence, but not necessarily the analysis-worthy temporal boundary.

### 4.8 E6: Boundary-Aware Scoring

#### Goal

Test whether candidate windows can be scored by comparing inside evidence to boundary/outside evidence, rather than only scoring relevance inside the window.

#### Method

E6 uses boundary-aware features built by `m0-graph-lite/build_boundary_aware_features.py` and utilities in `boundary_signal_utils.py`. It uses:

- `inside_relevance_mean`
- `inside_coherence`
- `boundary_contrast`
- `trend_coherence`
- `trend_signal_mean`
- `length_fit`

The scoring function `score_candidate_e6_boundary_aware` exists in `m0-exp/scoring_policies.py`.

#### Result

The user-provided current result list records E6 acceptable accuracy as 0.0000. The CSV artifacts most prominently summarize E6b rather than a stable E6 result. Therefore, E6's 0.0000 should be treated as a failure signal but not overinterpreted without inspecting whether there was an implementation, parsing, or feature-join issue.

#### Interpretation

The E6 idea is meaningful, but the observed result does not support its effectiveness. It motivates E6b's length-controlled and directional boundary scoring.

### 4.9 E6b: Boundary-Aware Scoring with Length Control

#### Goal

Fix E6's weakness by adding length control, stronger boundary contrast weighting, and directional adjustment.

#### Method

E6b uses `score_candidate_e6b_boundary_length_controlled` in `m0-exp/scoring_policies.py` and is reproduced in `m0_final/scoring.py` as `score_e6b_raw_boundary`. The formula is:

```text
base_score =
  0.15 * intent_fit
+ 0.15 * inside_relevance_mean
+ 0.10 * inside_coherence
+ 0.30 * boundary_contrast
+ 0.15 * trend_coherence
+ 0.15 * length_component

final_score = base_score + directional_adjustment - long_window_penalty
```

#### Result

In exploratory output, E6b has 0.10 acceptable accuracy for A and 0.12 for B, combined 0.11. It improves over E5b by 0.02 but remains below E2 by 0.08. The E6b diagnostics provided in the project context include:

- average overlap: 0.5480
- mean boundary error: 2.2950
- over-extension: 3.4800
- under-extension: 1.1100
- average boundary contrast: 0.4089
- average trend coherence: 0.8351
- average candidate length ratio: 0.4941
- average long-window penalty: 0.0130

The `e6b_comparison.csv` split view shows over-extension 3.92 for A and 3.04 for B.

#### Interpretation

E6b has signal: its overlap is not low, and boundary/trend features are nonzero. But acceptable accuracy remains limited because selected windows often over-extend. Trend coherence is high and long-window penalty is small, allowing long coherent windows to win even when boundaries are too broad.

### 4.10 E7 and E7b: Qualified Evidence Basis

#### Goal

Test whether E6b's weakness is partly caused by polluted evidence features. Instead of changing the scorer first, E7 changes which evidence chunks are allowed to influence feature construction.

#### Method

E7 uses:

- `m0-graph-lite/build_e7_chunk_qualifications.py`
- `m0-graph-lite/build_e7_boundary_aware_features.py`

The code uses build labels such as `e7b_rule_v1` and `e7b_boundary_aware_qualification_v1`. In the final outputs, the experiment is labeled `E7_qualified_boundary_scoring`, while the build version in `m0_final/config.py` is `BUILD_VERSION_E7 = "e7b_boundary_aware_qualification_v1"`. Therefore, "E7b" is best understood as the qualification/build-version implementation behind the E7 qualified-feature experiment, not as a separate fully reported experiment with an independent summary table.

E7 recomputes E6b-style features using qualified/routed evidence:

- `qualified_inside_relevance`
- `qualified_evidence_year_coverage`
- `qualified_boundary_contrast`
- `qualified_trend_coherence`
- `background_noise_ratio`

The conceptual shift is feature-specific routing: a chunk may be relevant for inside relevance but not for boundary contrast; general temporal evidence should not automatically become boundary evidence.

#### Result

In exploratory output, E7 has 0.12 acceptable accuracy for A and 0.18 for B, combined 0.15. It improves over E6b by 0.04 but remains below E2 by 0.04. In the final consolidated run, E7 has 0.14 acceptable accuracy, average overlap 0.5192, average boundary error 2.495, over-extension 3.41, and under-extension 1.58.

#### Interpretation

E7 supports the idea that evidence qualification improves boundary-aware scoring. However, it still does not surpass E2. Qualified evidence alone is not as stable as the E2 temporal prior.

### 4.11 FINAL: Prior-Guided Qualified Boundary Scoring

#### Goal

Combine E2's stable temporal prior with E7's qualified evidence basis and E6b's boundary features. This is the final consolidated method in `m0_final`, not a random additional ablation.

#### Method

Implemented in `m0_final/scoring.py` as `score_final_fused`. It computes:

```text
stable_prior =
  0.40 * intent_fit
+ 0.35 * length_fit
+ 0.25 * window_type_prior

qualified_boundary_score =
  0.25 * qualified_inside_relevance
+ 0.25 * qualified_evidence_year_coverage
+ 0.30 * qualified_boundary_contrast
+ 0.20 * qualified_trend_coherence
- 0.15 * background_noise_ratio

final_score =
  0.55 * stable_prior
+ 0.35 * qualified_boundary_score
+ 0.10 * directional_adjustment
- long_window_penalty
```

It also implements transparent gates:

- cap high scores when qualified evidence coverage is below threshold;
- halve trend coherence contribution when boundary contrast is low;
- add extra long-window penalty when candidate length ratio is high and qualified coverage is low.

#### Result

In `m0_final/outputs/csv/final_experiment_summary.csv`, FINAL has acceptable accuracy 0.17. It improves over E6b (0.09) and E7 (0.14) in the consolidated run, but it remains below E2 (0.19).

#### Interpretation

The implemented fusion partially recovers the stability lost in boundary/evidence-only methods. However, it does not establish FINAL as the best method. The supported conclusion is that combining stable prior and qualified evidence is promising but not yet sufficient to beat the simple E2 baseline.

## 5. Current Architecture

### 5.1 M0: Time Range Inference

#### Inputs

The implemented M0 scoring framework consumes:

- `rewritten_query` / `query_text`
- `target_subject` / `ticker`
- upstream candidate windows
- candidate window metadata (`start_year`, `end_year`, `window_type`)
- evidence probes and/or graph-lite features
- optional qualified boundary features

The generalized adapter in `m0_final/feature_loader.py` adds framework-level columns such as `target_subject`, `query_id`, `candidate_scope_id`, `start_time_unit`, `end_time_unit`, `scope_type`, and `predicted_temporal_scope` while preserving Bridge-specific fields.

#### Outputs

M0 outputs:

- selected candidate temporal scope
- `predicted_range` / `predicted_temporal_scope`
- score
- score components
- evaluation metrics after selection

The final outputs are:

- `m0_final/outputs/csv/final_experiment_runs.csv`
- `m0_final/outputs/csv/final_experiment_metrics.csv`
- `m0_final/outputs/csv/final_experiment_summary.csv`
- `m0_final/outputs/csv/final_method_comparison.csv`
- `m0_final/outputs/markdown/final_m0_report.md`

#### Logic

The M0 logic is candidate scoring and selection. Candidate generation is performed upstream by the M0 agent and stored in runtime tables. The scoring experiments select among those candidates. No gold labels are used for selection except in the non-deployable oracle E1.

The implemented method families are:

- E2 stable temporal prior
- E6b raw boundary scoring
- E7 qualified boundary scoring
- FINAL prior-guided qualified boundary scoring

#### Prompts

The current repository includes prompts for downstream answer generation and judging in `exp_original_m0_m3/prompts/`, not for M0 scoring itself. M0 scoring policies are deterministic Python functions. The upstream M0 agent scripts exist in `scripts/run_m0_agent.py` and `scripts/run_m0_agent_batch.py`, but this report does not reconstruct their full prompt behavior beyond noting that the upstream agent produces candidate windows.

#### Evaluation

M0 intrinsic evaluation uses:

- acceptable accuracy
- exact match
- overlap score
- start boundary error
- end boundary error
- mean boundary error
- predicted length
- over-extension
- under-extension

These metrics are implemented in `m0-exp/evaluation_utils.py` and `m0_final/evaluator.py`.

#### Current Status

M0 is the most complete part of the project. It has diagnostic, exploratory, graph-lite, qualified-evidence, and final consolidated experiment folders. The strongest current intrinsic method remains E2 at 0.19 acceptable accuracy; FINAL is close at 0.17 but not better.

### 5.2 M1: Temporal Retrieval

M1 is partially implemented in `exp_original_m0_m3/scripts/01_scope_constrained_bm25_retrieval.py`.

The implemented retrieval method is BM25-like lexical retrieval with a TF-IDF fallback if `rank_bm25` is unavailable. It compares:

- `NoScope`: no temporal filter
- `E2`: filter evidence to the E2 predicted temporal scope
- `FINAL`: filter evidence to the FINAL predicted temporal scope

The evidence source is `data/content/evidence_chunks.jsonl`. The loader streams JSONL and maps:

- `chunk_id` to `evidence_unit_id`
- `document_year` to `time_unit`
- `chunk_text` to `evidence_text`
- `mentioned_tickers` to analysis focus matching
- `doc_id` to `source_doc_id`

The current retrieval output `exp_original_m0_m3/outputs/retrieved_evidence/retrieved_evidence.csv` has 3,000 rows: 100 queries x 3 methods x 10 evidence units. There are no empty retrieval cases in the latest run.

M1 does not currently implement dense retrieval, reranking, temporal graph retrieval, or TA-RAG-style aggregation beyond scope filtering and BM25 retrieval.

### 5.3 M2: Evidence Structuring

M2 is implemented as deterministic timeline context formatting in `exp_original_m0_m3/scripts/02_build_timeline_context.py`.

It groups retrieved evidence by `time_unit`, sorts time units ascending, sorts evidence within a time unit by retrieval rank, truncates each evidence text, and enforces a maximum context length. The timeline format is:

```text
[Time: 2018]
- Evidence text...
- Evidence text...

[Time: 2019]
- Evidence text...
```

The current output `exp_original_m0_m3/outputs/timeline_contexts/timeline_contexts.csv` has 300 rows. It has no empty contexts. However, it has only 52 unique timeline contexts across 300 rows, and E2 vs FINAL contexts are identical for 63 of 100 queries. This is a limitation for downstream comparison because the answer model often sees the same context for E2 and FINAL.

M2 does not currently implement learned temporal evidence structuring, event extraction, phase segmentation, or temporal graph construction.

### 5.4 M3: Answer Generation

M3 is implemented but not fully run in the current output state.

#### Implemented

`exp_original_m0_m3/scripts/03_generate_answers_qwen7b.py` uses Ollama to call `qwen2.5:7b`. It reads `timeline_contexts.jsonl`, uses `exp_original_m0_m3/prompts/answer_generation_prompt.txt`, and outputs:

- `generated_answers.jsonl`
- `generated_answers.csv`

The prompt instructs the model to:

- answer using only provided evidence;
- avoid outside knowledge;
- say "insufficient evidence" when evidence is insufficient;
- mention supporting time periods;
- return JSON only.

#### Current Output State

The current `generated_answers.csv` contains one row only, produced by a smoke test. The `generated_answer` field is empty because the model output was truncated with a very small `num_predict` setting. Therefore no full M3 answer generation results are available for the 100-query comparison.

#### Proposed

The intended downstream experiment is to run Qwen 7B for all 300 timeline contexts and compare NoScope, E2, and FINAL under fixed retrieval/context/prompt/model settings. This has not yet been completed in the outputs.

## 6. Evaluation Framework

### 6.1 Intrinsic Evaluation

Intrinsic M0 evaluation measures whether the selected temporal scope matches the target temporal scope.

#### Overlap Score

Overlap is inclusive intersection-over-union over years:

```text
overlap = |predicted_years intersect gold_years| / |predicted_years union gold_years|
```

It captures partial correctness even when exact boundaries differ.

#### Boundary Error

Boundary error includes:

- `start_boundary_error = abs(predicted_start - gold_start)`
- `end_boundary_error = abs(predicted_end - gold_end)`
- `mean_boundary_error = average of start and end errors`

Boundary error matters because a window can have good overlap but still start or end at the wrong time.

#### Acceptable Accuracy

Acceptable accuracy is true if the predicted range exactly equals one of the annotated acceptable ranges. It is strict but reflects the project goal: infer a suitable temporal scope, not merely retrieve some overlapping years.

#### Over-Extension and Under-Extension

Over-extension measures extra years included outside the gold range. Under-extension measures missing gold years. These diagnostics are important because E6b often has reasonable overlap but over-extends.

### 6.2 Extrinsic Evaluation

The extrinsic evaluation folder `exp_original_m0_m3/` is designed to compare:

- `NoScope`: BM25 over all relevant time-indexed evidence without temporal filtering.
- `E2`: BM25 constrained to E2 predicted temporal scope.
- `FINAL`: BM25 constrained to FINAL predicted temporal scope.
- optional E7, not included in the main comparison by default.

The planned evaluation tests whether M0 scope selection improves downstream answer generation. However, only retrieval and context construction have been completed in the current output state. Answer generation, LLM judging, and RAGAS evaluation are not complete for the full 300-row experiment.

### 6.3 LLM Judge Metrics

`exp_original_m0_m3/scripts/04_judge_answers_qwen14b.py` implements Qwen 14B judging with these dimensions:

- `answer_correctness`
- `faithfulness_to_evidence`
- `temporal_coverage`
- `temporal_scope_usefulness`

The judge prompt is in `exp_original_m0_m3/prompts/judge_prompt.txt`. The overall score is computed as:

```text
overall_score =
  0.40 * answer_correctness_score
+ 0.25 * faithfulness_score
+ 0.25 * temporal_coverage_score
+ 0.10 * temporal_scope_score
```

The current `judgments_flat.csv` contains only one row, so no reliable aggregate conclusion can be drawn.

### 6.4 RAGAS Metrics

`exp_original_m0_m3/scripts/04b_ragas_evaluate_answers.py` was added as a supplementary evaluator. It maps:

- `question = query_text`
- `answer = generated_answer`
- `ground_truth/reference = reference_answer`
- `contexts = retrieved_contexts` if present, else `[timeline_context]`

It attempts to evaluate:

- `faithfulness`
- `answer_correctness`
- `context_precision`
- `context_recall`

The script handles missing RAGAS/LangChain/Ollama support by writing `ragas_error` instead of crashing. The current RAGAS output has one row and reports `generated_answer is empty; skipped RAGAS evaluation`. Therefore RAGAS has not yet produced usable scores.

## 7. Failure Analysis

### 7.1 Type A Failures

In this project, Type A corresponds to broad/general rewritten queries that often lack explicit temporal direction. Examples include "How did AGG's stock price generally trend?" The model must infer whether the relevant period is early, late, full, or a multi-year interval from weak signals.

Observed patterns:

- E0 often selects rolling_8y or broad windows.
- E2 improves but strongly prefers rolling_8y.
- E6b often selects long prefix/suffix windows.
- FINAL reduces but does not eliminate over-extension.

Evidence:

- `e2_vs_e3_summary.csv` reports E2 rolling_8y selection rate 0.91.
- `m0_final` reports E2 predicted length 7.73 and FINAL predicted length 7.79, while gold lengths average around 6.24 in exploratory outputs.
- E6b over-extension is 3.48 in the diagnostic values and 3.36 in final output.

Interpretation:

Type A broad queries require phase or boundary evidence that is not reliably captured by current features. Stable priors help but can lock onto generic rolling windows.

### 7.2 Type B Failures

Type B often contains vague relative cues such as "earlier periods", "later periods", or "over time". The problem is not detecting the words; the problem is translating vague direction into the correct temporal boundary.

Observed patterns:

- Earlier/later wording can still produce windows that are too broad or shifted.
- Some later-period queries are predicted as early windows in diagnostic outputs.
- Directional features improve individual examples but do not produce robust aggregate gains.

Evidence:

The diagnostic report lists Type B error cases such as:

- query: "How did BIDU's stock price generally trend in later periods?"
- predicted range: `[2012, 2017]`
- gold range: `[2020, 2022]`
- error note: `predicted_too_long`
- overlap: 0.0
- mean boundary error: 6.5

E3 adds directional/position awareness but lowers combined acceptable accuracy from 0.19 to 0.16. E4 and E5 do not fix this at aggregate level.

Interpretation:

Vague relative language needs stronger anchor resolution. The current rules detect "earlier" or "later" but do not robustly infer where the anchor should fall when the explicit anchor has been removed.

### 7.3 Type C Failures

The repository does not contain a formal implemented Type C taxonomy equivalent to Type A/B rewrite types. For this report, Type C is used cautiously to refer to content/evidence-induced failures: cases where temporal scope scoring is distorted by retrieval or evidence feature construction rather than by query wording alone.

Observed patterns:

- Query relevance features can become dense and nondiscriminative.
- Boundary-aware features can be polluted by background chunks.
- Downstream retrieval can produce duplicate or highly overlapping timeline contexts.

Evidence:

- E5 query-relevant year ratio is often 1.0 in report tables, while accuracy is low.
- E6b has high trend coherence but low acceptable accuracy.
- E7 improves over E6b, supporting the pollution hypothesis.
- M2 timeline contexts currently have 52 unique contexts across 300 rows, with 63 identical E2/FINAL contexts.

Interpretation:

Evidence signals contain useful information but must be role-qualified. Raw semantic relevance is not enough to define temporal boundaries.

### 7.4 Over-Extension

Over-extension is the clearest recurring failure mode. E6b often finds a relevant region but includes too many years. In the final consolidated run:

- E6b over-extension: 3.36
- E7 over-extension: 3.41
- FINAL over-extension: 2.89
- E2 over-extension: 2.81

FINAL reduces over-extension relative to E6b/E7 but does not beat E2.

### 7.5 Under-Extension

E5 and E5b tend toward shorter windows in some outputs. In `experiment_summary.csv`, E5 and E5b have average predicted lengths around 5.0 while gold length is 6.24, and under-extension values are higher than E2.

### 7.6 Wrong Anchor

Wrong-anchor failures occur when a later or earlier query is mapped to the opposite or overly broad side of the timeline. These are visible in diagnostic Type B examples and oracle gap tables.

### 7.7 Insufficient Evidence

The original dataset coverage diagnostic reports full gold coverage, so insufficient corpus coverage is not the primary M0 intrinsic failure in the current dataset. However, answer generation may still experience insufficient retrieved evidence because M1 retrieval can select top-ranked chunks that do not cover the necessary years or phases. This has not yet been evaluated at full scale.

### 7.8 Retrieval Collapse

The current downstream retrieval/context stage shows context overlap: 52 unique contexts for 300 rows, and 63 identical E2/FINAL contexts. This does not mean retrieval failed, because no context is empty and NoScope differs from scoped methods. But it limits the power of downstream comparison: if E2 and FINAL often produce the same context, answer quality differences will be small or unobservable.

## 8. Research Conclusions Supported Today

### 8.1 Supported

1. Candidate generation is not the main bottleneck for the evaluated M0 sample. Oracle selection reaches acceptable accuracy 1.0, while deployable methods remain far lower.

2. Stable temporal-prior scoring improves over the original runtime baseline. E2 improves acceptable accuracy from E0's 0.08 to 0.19.

3. The E2 prior is strong but biased. It has high rolling_8y selection rate (0.91), indicating reliance on a stable window prior.

4. Adding temporal position awareness can reduce rolling_8y bias but does not automatically improve accuracy. E3 lowers rolling_8y selection rate to 0.68 but drops accuracy to 0.16.

5. Graph-lite continuity and density features, as implemented, do not improve over E2. E4 has 0.12 acceptable accuracy.

6. Query relevance and hybrid query relevance, as implemented, do not solve temporal boundary inference. E5 and E5b remain at 0.08 and 0.09 combined acceptable accuracy.

7. Boundary-aware features contain signal but suffer from over-extension. E6b improves over E5b but remains below E2 and often selects windows that are too long.

8. Chunk-level evidence qualification improves boundary-aware scoring. E7 improves over E6b in exploratory and final comparisons.

9. Fusing stable prior with qualified boundary features helps relative to E6b/E7 in the final run, but does not beat E2. FINAL reaches 0.17 while E2 reaches 0.19.

10. ECT-QA can naturally enter the framework for downstream answer evaluation or qualitative feasibility, but not for M0 acceptable-accuracy evaluation without reliable gold temporal scopes.

### 8.2 Not Yet Supported

1. It is not yet supported that FINAL is the best M0 method. It is below E2 in the final intrinsic output.

2. It is not yet supported that temporal scope inference improves downstream answer quality. The M0-M3 downstream pipeline is implemented through retrieval/context construction, but full answer generation and judging have not been completed.

3. It is not yet supported that temporal graph construction improves diachronic QA. Graph-lite features were tried and did not improve M0; full temporal evidence graphs are not implemented.

4. It is not yet supported that RAGAS validates the pipeline. The current RAGAS output is a skipped one-row smoke test due to empty generated answer.

5. It is not yet supported that ECT-QA provides quantitative M0 scope accuracy, because no reliable gold temporal scope is derivable from current inspection.

## 9. Open Problems

1. Event boundary inference remains unresolved. Current features do not reliably determine when the analysis-worthy trend begins or ends.

2. Phase detection is not implemented. The system does not segment a timeline into rising, falling, volatile, recovery, or transition phases before selecting scope.

3. Anchor recovery for implicit queries remains weak. When explicit "before 2020" or "after 2019" labels are removed, current rules often cannot recover the hidden anchor.

4. Evidence role qualification is only rule-based. E7/E7b improves boundary features, but qualification quality depends on heuristic routing.

5. Long coherent windows can still win. E6b and E7 show that trend coherence can reward long windows unless penalties and priors are calibrated.

6. Downstream answer evaluation is incomplete. The pipeline needs a full run of answer generation, LLM judging, RAGAS, and summary aggregation before extrinsic claims can be made.

7. Retrieval diversity and deduplication need improvement. Timeline contexts are highly duplicated, especially between E2 and FINAL.

8. Dataset generalization remains open. ECT-QA feasibility is promising, but quantitative M0 evaluation requires gold or supporting temporal references.

9. The relationship between acceptable accuracy and answer quality is unknown. A predicted scope can be unacceptable by strict boundary criteria but still produce adequate evidence for an answer, or vice versa.

10. Financial-domain assumptions may leak into features. `m0_final` adds generalized vocabulary, but the original benchmark and evidence corpus remain financial.

## 10. Future Framework (Design Only)

```text
NOT IMPLEMENTED
```

This section describes design ideas only. These are not demonstrated contributions and should not be presented as implemented results.

### 10.1 Graph-Assisted Temporal Scope Inference

The design rationale is that temporal scope inference may benefit from a structured representation of entities, events, metrics, and time-indexed evidence. A graph-assisted system could represent:

- target subject nodes;
- time-unit nodes;
- evidence-unit nodes;
- event or phase nodes;
- metric/state nodes;
- relations such as supports, precedes, contrasts, begins, ends, and shifts.

The current repository contains graph-lite feature extraction, but not a full temporal evidence graph. E4 graph-lite did not improve over E2, so any future graph approach must address why current graph-lite signals were not discriminative.

### 10.2 Event / Phase / Turning Point Representation

The current M0 features score candidate windows but do not explicitly model phases. A future design could detect:

- stable periods;
- growth phases;
- decline phases;
- volatility phases;
- turning points;
- recovery periods;
- pre/post-anchor transitions.

Such a design could help distinguish a long coherent region from the more precise analysis-worthy span. This is especially relevant for Type A broad queries and Type B later/earlier queries. However, no event/phase extraction pipeline is currently implemented.

### 10.3 Temporal Evidence Graph

A temporal evidence graph could connect evidence units to time units and roles:

- inside evidence;
- boundary evidence;
- background evidence;
- trend evidence;
- transition evidence;
- contradiction or contrast evidence.

E7/E7b is a partial step toward this because it qualifies chunks for feature-specific roles. But it does not construct a graph of temporal events or phases. The implemented E7 qualification should not be overstated as a full temporal graph.

### 10.4 Prior-Guided Qualified Evidence with Learned Calibration

FINAL currently uses transparent heuristic weights and gates. A future design could learn or tune weights using held-out validation, but this has not been implemented in a rigorous way. Any learned calibration would need to preserve the rule that gold labels cannot be used for candidate selection in test examples.

### 10.5 Full M0-M3 Temporal RAG Evaluation

The intended downstream experiment is:

```text
M0 predicted scope
-> scope-constrained BM25 retrieval
-> time-ordered timeline context
-> Qwen 7B answer generation
-> Qwen 14B temporal judge and RAGAS supplementary evaluation
```

The scripts exist, but the full run is not complete in current outputs. Therefore, this remains a near-term experimental plan rather than a completed result.

## Appendix A. Evidence Inventory

Key implemented folders:

- `m0-diagnostic/`: diagnostic analysis showing candidate generation is not the primary bottleneck.
- `m0-exp/`: exploratory E0-E7 scoring experiments.
- `m0-graph-lite/`: graph-lite, query relevance, hybrid relevance, boundary features, and E7/E7b qualification feature builders.
- `m0_final/`: consolidated final M0 comparison among E2, E6b, E7, and FINAL.
- `exp_ect_qa/`: external dataset feasibility/adaptation experiment.
- `exp_original_m0_m3/`: downstream validation scaffold for original dataset.

Key outputs:

- `m0-exp/outputs/csv/experiment_summary.csv`
- `m0-exp/outputs/csv/e2_vs_e3_summary.csv`
- `m0-exp/outputs/csv/e6b_comparison.csv`
- `m0-exp/outputs/csv/e7_filtered_e6b_comparison.csv`
- `m0_final/outputs/csv/final_experiment_summary.csv`
- `exp_ect_qa/reports/schema_inspection_report.md`
- `exp_ect_qa/reports/ectqa_m0_experiment_report.md`
- `exp_original_m0_m3/outputs/scope_runs/scope_runs.csv`
- `exp_original_m0_m3/outputs/retrieved_evidence/retrieved_evidence.csv`
- `exp_original_m0_m3/outputs/timeline_contexts/timeline_contexts.csv`

Key caution:

Any future paper should distinguish clearly between intrinsic M0 results, partially completed downstream pipeline artifacts, and future graph/phase design ideas.

## Appendix B. Detailed Artifact Notes

This appendix records additional implementation evidence that may be useful when transforming this internal report into a paper. It is intentionally descriptive rather than argumentative.

### B.1 Database and Export Layer

The repository includes PostgreSQL schema initialization files under `postgres/init/`. These files create and index schemas for content, QA, runtime outputs, and debug views. The presence of files such as `001_extensions_and_schemas.sql`, `002_schema_qa.sql`, `003_schema_content.sql`, `006_schema_runtime.sql`, and `008_views_m0_debug.sql` indicates that the project originally used PostgreSQL as the main experiment store. Later folders export tables to CSV and JSONL, allowing local analysis without a live database.

The core export scripts are:

- `scripts/export_qa_tables.py`
- `scripts/export_content_tables.py`
- `scripts/export_runtime_tables.py`
- `scripts/build_qa_content_coverage.py`

The final downstream folder was adapted to read from these exported local files rather than requiring a database connection. This matters for reproducibility: the M0 final scoring code can still read database tables, but the M0-M3 validation scaffold can operate from `data/` exports.

### B.2 Upstream M0 Agent and Candidate Generation

The repository contains `scripts/run_m0_agent.py`, `scripts/run_m0_agent_batch.py`, and `scripts/m0_agent_tools.py`. The current report does not claim a detailed agent architecture because the strongest experiment artifacts evaluate the candidate windows after they have already been generated. However, the diagnostic results show that the upstream candidate generator produces many candidates per run. In `m0-diagnostic/outputs/markdown/m0_diagnostic_report.md`, the candidate oracle summary reports `avg_total_candidates = 43.0` for both rewrite type A and rewrite type B under `query_with_corpus_metadata`.

The candidate window types visible in outputs include:

- `rolling_8y`
- `rolling_5y`
- `rolling_3y`
- `prefix_window`
- `suffix_window`
- `early_half`
- `late_half`
- `recent_5y`
- `full_entity_range`
- `content_supported_window`
- `verification_candidate`

Not every method selects every type. For example, E2 heavily selects `rolling_8y`; E6b tends to select prefix/suffix windows; E7 shifts toward qualified prefix/suffix and occasional rolling windows. The oracle selects a mix of prefix, suffix, rolling, early-half, recent, and full-range candidates, indicating that no single window type is universally correct.

### B.3 M0 Scoring Implementation Notes

The exploratory scorer lives in `m0-exp/scoring_policies.py`. The consolidated final scorer lives in `m0_final/scoring.py`. The final scorer intentionally reproduces the key deployed comparison methods rather than preserving every exploratory experiment:

- `score_e2_prior`
- `score_e6b_raw_boundary`
- `score_e7_qualified_boundary`
- `score_final_fused`
- `select_best_candidate`

The final scorer uses deterministic tie-breaking. For FINAL, the tie-break prefers higher stable prior, higher qualified boundary contrast, higher qualified evidence coverage, lower candidate length ratio, higher qualified trend coherence, and then deterministic candidate id. This design is implemented in `select_best_candidate` and matters because close-score candidates are common. It is not a learned reranker.

The final scoring constants are in `m0_final/config.py`. The build versions recorded there include:

- `BUILD_VERSION_E6B = "boundary_v1"`
- `BUILD_VERSION_E7 = "e7b_boundary_aware_qualification_v1"`
- `BUILD_VERSION_FINAL = "final_prior_qualified_boundary_v1"`

This naming should be handled carefully in future writing. The experiment table uses "E7" as the method name, while the feature build version uses "e7b". The safe description is: E7 is the qualified boundary scoring experiment; E7b refers to the implemented rule/build-version of the qualification feature construction.

### B.4 Feature Construction Layers

The graph-lite folder is organized as a sequence of feature builders:

- `build_graph_lite_features.py` creates candidate graph features based on temporal structure.
- `build_query_relevant_graph_features.py` builds query-relevant graph features.
- `build_hybrid_query_relevant_graph_features.py` adds hybrid lexical/dense relevance.
- `build_boundary_aware_features.py` builds E6/E6b boundary-aware features.
- `build_e7_chunk_qualifications.py` qualifies chunks for feature-specific roles.
- `build_e7_boundary_aware_features.py` recomputes boundary features using the qualified evidence basis.

The CSV outputs under `m0-graph-lite/outputs/csv/` preserve each stage. Important files include:

- `candidate_graph_features.csv`
- `candidate_query_graph_features.csv`
- `hybrid_candidate_query_graph_features.csv`
- `candidate_boundary_features.csv`
- `e7_chunk_temporal_qualifications.csv`
- `e7_candidate_boundary_features.csv`
- `e7_candidate_boundary_diagnostics.csv`
- `e7_candidate_boundary_diagnostic_summary.csv`

These files support the interpretation that the project progressively moved from simple priors to graph-lite structure, then query relevance, then boundary-aware features, then qualified evidence.

### B.5 Final M0 Schema and Vocabulary Generalization

`m0_final/README.md` states that the final folder implements a "Prior-guided Qualified Evidence Temporal Scope Inference Framework" and adds a dataset-neutral vocabulary. The generalized pipeline is:

```text
Query
-> Subject Anchor Extraction
-> Temporal Intent Understanding
-> Candidate Temporal Scope Generation
-> Time-indexed Evidence Retrieval / Probing
-> Evidence Role Qualification
-> Temporal Feature Construction
-> Prior-guided Qualified Evidence Scoring
-> Predicted Temporal Scope
```

The current code preserves old Bridge columns while adding framework-level aliases:

- `ticker` / `company` -> `target_subject`
- `rewritten_query` / `query` -> `query_text`
- `bridge_id` / `m0_input_id` -> `query_id`
- `candidate_window_id` -> `candidate_scope_id`
- `start_year` / `end_year` -> `start_time_unit` / `end_time_unit`
- `window_type` -> `scope_type`
- `gold_range` -> `gold_temporal_scope`
- `acceptable_ranges` -> `acceptable_temporal_scopes`

This is an adapter layer, not a destructive schema migration. It supports external dataset adaptation while preserving Bridge compatibility.

### B.6 ECT-QA Feasibility Artifacts

The ECT-QA experiment folder includes scripts from import through feature construction and m0_final scoring reuse. The schema inspection report identifies two Hugging Face configs: `questions` and `corpus`. The `questions_train` split has 1,105 rows, while `corpus_train` has 480 rows. The corpus includes company name, stock code, sector, year, quarter, URL, raw content, cleaned content, and token count.

The schema summary records:

- `has_query_text = true`
- `has_analysis_focus_source = true`
- `analysis_focus_source_type = metadata`
- `has_time_indexed_evidence = true`
- `has_evidence_units = true`
- `has_reference_answer = true`
- `has_supporting_evidence_dates = false`
- `can_derive_gold_temporal_scope = false`
- `usable_for_m0_scope_eval = false`
- `usable_for_downstream_answer_eval = true`

The ECT-QA M0 experiment report explicitly states that no reliable gold temporal scope was available, so M0 acceptable accuracy was not computed. It reports only diagnostic average scores for 744 scored examples:

| Experiment | n | avg_score |
| --- | ---: | ---: |
| E2 | 744 | 0.4369 |
| E6b | 744 | 0.2522 |
| E7 | 744 | 0.2522 |
| FINAL | 744 | 0.3081 |

These are not accuracy scores. A future paper should not compare them directly to Bridge acceptable accuracy.

### B.7 Downstream M0-M3 Artifact State

The downstream experiment folder is currently a scaffold plus partial outputs. The implemented scripts are:

- `00_prepare_scope_runs.py`
- `01_scope_constrained_bm25_retrieval.py`
- `02_build_timeline_context.py`
- `03_generate_answers_qwen7b.py`
- `04_judge_answers_qwen14b.py`
- `04b_ragas_evaluate_answers.py`
- `05_summarize_m0_m3_results.py`

The current complete downstream artifacts are:

- `scope_runs.csv`: 300 rows, with reference answers and gold temporal scopes filled for all rows.
- `retrieved_evidence.csv`: 3,000 rows, with 1,000 rows for each of NoScope, E2, and FINAL.
- `timeline_contexts.csv`: 300 rows, no empty contexts.

The current incomplete downstream artifacts are:

- `generated_answers.csv`: 1 row only, from smoke testing.
- `judgments_flat.csv`: 1 row only.
- `ragas_scores_flat.csv`: 1 row only, skipped because `generated_answer` is empty.

The downstream scripts therefore demonstrate pipeline wiring, not downstream effectiveness. A full run must regenerate answers for all 300 contexts and rerun both judge and RAGAS evaluation before any answer-quality claim is valid.

### B.8 Prompt Artifacts

The answer generation prompt instructs the model to answer using only the provided time-ordered evidence, avoid outside knowledge, state "insufficient evidence" if needed, mention supporting time periods, and return JSON only. This prompt is intentionally method-agnostic: it does not tell the answer model whether the evidence came from NoScope, E2, or FINAL.

The judge prompt asks Qwen 14B to evaluate correctness against the reference answer, faithfulness to evidence, temporal coverage, and temporal scope usefulness. It also does not reveal the scope method name in the prompt. This is important for fair downstream comparison.

The RAGAS script is supplementary and explicitly documented as general RAG evaluation, not temporal-specific evaluation. Temporal coverage and temporal scope usefulness remain outside standard RAGAS metrics and are handled by the temporal judge script.

## Appendix C. Implementation Status Matrix

| Component | Implemented? | Evidence | Current Status |
| --- | --- | --- | --- |
| Original QA construction | Yes | `data/qa/*.jsonl`, `qa_quality_report.csv` | Completed exports exist. |
| M0 candidate generation | Yes | runtime exports, M0 diagnostic reports | Candidate generation not main bottleneck in 100-query diagnostic. |
| M0 intrinsic scoring experiments E0-E7 | Yes | `m0-exp/outputs/` | Completed exploratory comparison. |
| M0 final comparison E2/E6b/E7/FINAL | Yes | `m0_final/outputs/csv/final_experiment_summary.csv` | Completed final intrinsic comparison. |
| Dataset-neutral vocabulary adapter | Yes | `m0_final/feature_loader.py`, `schemas.py`, `README.md` | Added without destructive renaming. |
| ECT-QA import and feasibility | Yes | `exp_ect_qa/processed/`, `reports/` | Feasible for downstream validation, not M0 accuracy. |
| M1 scope-constrained retrieval | Yes | `retrieved_evidence.csv` | 3000 rows generated for NoScope/E2/FINAL. |
| M2 timeline context construction | Yes | `timeline_contexts.csv` | 300 contexts generated; context duplication remains. |
| M3 answer generation | Partially | `03_generate_answers_qwen7b.py` | Script exists; full 300-row output not currently present. |
| Qwen 14B judging | Partially | `04_judge_answers_qwen14b.py` | Script exists; current output is one-row smoke test. |
| RAGAS evaluation | Partially | `04b_ragas_evaluate_answers.py` | Script exists; current output is one-row skipped evaluation. |
| Full downstream result summary | No | `05_summarize_m0_m3_results.py` | Requires complete generated/judged outputs. |
| Temporal evidence graph | No | Future design only | Not implemented. |
| Event/phase segmentation | No | Future design only | Not implemented. |
| Learned calibration | No | Future design only | Not implemented. |
