# Bridge Rewrite Dataset v3 M0 Experiment

This folder runs an M0-only temporal scope inference experiment for `bridge_rewrite_dataset_v3.csv`.

The goal is to compare three implicit query types:

- Type L: latent / no time clue
- Type R: relative expression
- Type A: event-anchored

This experiment does not run M1-M3 retrieval, answer generation, RAGAS, or LLM judging.

## Run

```bash
python exp_bridge_v3_m0/scripts/00_prepare_bridge_v3_inputs.py \
  --input m0_final/bridge_rewrite_dataset_v3.csv \
  --output exp_bridge_v3_m0/data/bridge_v3_m0_inputs.csv

python exp_bridge_v3_m0/scripts/01_run_bridge_v3_m0.py \
  --input exp_bridge_v3_m0/data/bridge_v3_m0_inputs.csv \
  --output-dir exp_bridge_v3_m0/outputs/csv

python exp_bridge_v3_m0/scripts/02_summarize_bridge_v3_results.py \
  --runs exp_bridge_v3_m0/outputs/csv/bridge_v3_m0_runs.csv \
  --output-dir exp_bridge_v3_m0/outputs
```

Gold ranges are used only for evaluation. Candidate generation uses adapter-level dataset year bounds and anchor/event metadata, not per-query gold ranges as direct candidates.

## Outputs

- `outputs/csv/bridge_v3_m0_runs.csv`
- `outputs/csv/bridge_v3_m0_metrics.csv`
- `outputs/csv/bridge_v3_m0_summary_by_method.csv`
- `outputs/csv/bridge_v3_m0_summary_by_type.csv`
- `outputs/csv/bridge_v3_m0_error_cases.csv`
- `outputs/markdown/bridge_v3_m0_report.md`

## Boundary Features

The adapter runs E2 and FINAL. FINAL is diagnostic in this adapter because Bridge v3 does not include real E7 qualified boundary features. Neutral adapter-level feature values are supplied so the fused scorer can execute, and the report marks this limitation.
