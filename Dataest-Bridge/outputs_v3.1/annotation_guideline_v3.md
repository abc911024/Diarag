# Bridge Benchmark v3 Annotation Guideline

## Core principle
The pipeline separates annotation from rewrite generation to avoid temporal answer leakage.

## Annotation Layer
Metadata is allowed here because this layer creates evaluation labels:
- plot_time_bounds -> gold_start_date, gold_end_date, gold_range
- query_type_info / context_details_for_sampling -> temporal_scope_type, anchor_year, anchor_direction
- gold_range -> acceptable_ranges

These fields are evaluation targets and should not be given to the M0 predictor.

## Rewrite Layer
Rewrite generation is text-only. It may use only:
- original_query
- ticker
- stock_name

It must not use:
- plot_time_bounds
- gold_range
- gold_start_date / gold_end_date
- acceptable_ranges
- temporal_scope_type
- anchor_year / anchor_direction
- query_type_info
- context_details_for_sampling

## Rewrite Types
- Type A: remove all time expressions and produce a natural no-time query.
- Type B: use a vague temporal phrase derived from the original query text only.
- Type C: generate only when the original query contains a clear event anchor; otherwise mark not_applicable.

## Quality Flags
- contains_explicit_year
- contains_original_temporal_label
- keeps_entity
- keeps_diachronic_intent
- is_non_empty_question

A rewrite passes only when it is non-empty, contains no explicit year, does not retain the source temporal label, preserves the entity, and preserves diachronic intent.
