# ECT-QA Framework Feasibility Report

Recommended mode: **Mode D: Not usable**

## Can ECT-QA Naturally Enter the Framework?

ECT-QA can enter the framework only to the extent that its own fields or metadata support query text, analysis focus identification, time-indexed evidence units, and candidate temporal scopes. This report does not force Bridge-style company/ticker assumptions.

## Analysis Focus

- Source type: `metadata`
- Company/ticker optional fields present: `True`
- If explicit focus metadata is missing, the next step is rule-based query-derived focus extraction with confidence scoring.

## Available Components

- analysis_focus

## Missing or Uncertain Components

- query_text
- time-indexed evidence units
- gold temporal scope
- reference answer

## Gold Temporal Scope

Can derive gold temporal scope without inventing it: **False**.

If this is false, do not report M0 acceptable accuracy. Use qualitative scope validation or downstream answer validation instead.

## Recommended Next Steps

1. Run `03_identify_analysis_focus.py` to create query-level analysis focus candidates.
2. Build M0 inputs and evidence units only from fields actually present in ECT-QA.
3. Generate candidate temporal scopes from available time units, not from gold labels.
4. Run M0 scoring only for methods whose required features are available.
5. If gold temporal scope is unavailable, report predicted scope distributions and downstream answer feasibility rather than M0 accuracy.

## No Forced Conversion Assumptions

Company/ticker fields are optional. They are not required and should not be invented. Gold temporal scopes should never be invented.