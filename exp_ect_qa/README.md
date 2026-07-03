# ECT-QA Framework Validation Experiment

This folder validates whether the Hugging Face dataset `austinmyc/ECT-QA` can naturally enter the **Prior-guided Qualified Evidence Temporal Scope Inference Framework**.

It does not force ECT-QA into the Bridge schema. In particular, it does not require a company, ticker, or explicit `target_subject` field. If company or ticker metadata exists, it is treated only as one possible source for `analysis_focus`.

## Framework Question

The experiment asks whether ECT-QA can provide or support:

- `query_text`
- `analysis_focus`
- time-indexed `evidence_unit`s
- candidate temporal scopes
- optional gold/supporting temporal references for M0 evaluation
- optional reference answers for downstream answer validation

`analysis_focus` is required by the framework because temporal scope inference needs to know what temporal development is being analyzed. However, it may be inferred from metadata, question text, document or transcript metadata, or retrieval context. If no reliable focus can be identified, the example is marked low-confidence or unusable for framework validation.

## Pipeline

```text
Query
-> Analysis Focus Identification
-> Temporal Intent Understanding
-> Query/Focus-conditioned Time-indexed Evidence Unit Collection
-> Candidate Temporal Scope Generation
-> Evidence Role Qualification
-> Temporal Feature Construction
-> Prior-guided Temporal Scope Scoring
-> Predicted Temporal Scope
```

## Scripts

Run from the repository root.

1. Import ECT-QA:

```bash
python exp_ect_qa/scripts/00_import_ectqa.py
```

2. Inspect the schema:

```bash
python exp_ect_qa/scripts/01_inspect_ectqa_schema.py
```

3. Build a feasibility report:

```bash
python exp_ect_qa/scripts/02_build_framework_feasibility_report.py
```

4. Identify `analysis_focus`:

```bash
python exp_ect_qa/scripts/03_identify_analysis_focus.py
```

5. Build framework-level M0 inputs:

```bash
python exp_ect_qa/scripts/04_build_ectqa_m0_inputs.py
```

6. Build evidence units:

```bash
python exp_ect_qa/scripts/05_build_ectqa_evidence_units.py
```

7. Generate candidate temporal scopes:

```bash
python exp_ect_qa/scripts/06_generate_candidate_temporal_scopes.py
```

8. Build scope features:

```bash
python exp_ect_qa/scripts/07_build_scope_features.py
```

9. Optionally run M0 scoring methods on the processed CSVs:

```bash
python exp_ect_qa/scripts/08_run_m0_final_on_ectqa.py
```

## Evaluation Modes

M0 temporal scope evaluation is only possible if ECT-QA provides explicit or derivable supporting temporal references, such as supporting evidence dates or temporal metadata. If no reliable gold temporal scope exists, the experiment must not report M0 acceptable accuracy.

ECT-QA may still be useful for downstream answer validation if it provides reference answers and time-indexed evidence. This experiment does not claim to reproduce the original ECT-QA benchmark; it is an external feasibility and validation experiment for the temporal scope inference framework.
