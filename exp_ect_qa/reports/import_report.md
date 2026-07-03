# ECT-QA Import Report

- Dataset: `austinmyc/ECT-QA`
- `datasets` version: `4.8.5`
- `pandas` version: `3.0.2`

- Configs imported: `['questions', 'corpus']`

## Splits

## Questions Fallback Import

The Hugging Face `questions` config can fail because its JSON files do not share one fixed schema. The importer therefore downloads each question JSON file separately and preserves the union of columns.

### questions / global_questions_new

- Rows: 28
- Columns: ['question', 'role', 'type', '_hf_config', '_hf_split', '_question_file', '_row_id']

### questions / global_questions_old

- Rows: 72
- Columns: ['question', 'role', 'type', '_hf_config', '_hf_split', '_question_file', '_row_id']

### questions / local_questions_new

- Rows: 349
- Columns: ['question', 'answer', 'reasoning_type', 'question_type', 'num_hops', 'evidence_list', '_hf_config', '_hf_split', '_question_file', '_row_id']

### questions / local_questions_old

- Rows: 656
- Columns: ['question', 'answer', 'reasoning_type', 'question_type', 'num_hops', 'evidence_list', '_hf_config', '_hf_split', '_question_file', '_row_id']

### questions / combined train

- Rows: 1105
- Columns: ['question', 'role', 'type', '_hf_config', '_hf_split', '_question_file', '_row_id', 'answer', 'reasoning_type', 'question_type', 'num_hops', 'evidence_list']

First examples:

```json
[
  {
    "question": "How did Simon Property Group, Inc.’s strategic investments in platform businesses influence its funds from operations (FFO) growth trajectory and capital allocation priorities from 2021 Q1 to 2024 Q4?",
    "role": "Regulatory Compliance Officer",
    "type": "company_level_multi_time",
    "_hf_config": "questions",
    "_hf_split": "train",
    "_question_file": "global_questions_new",
    "_row_id": "global_questions_new_0"
  },
  {
    "question": "How did varying levels of macroeconomic uncertainty and consumer confidence in 2024 Q3 influence revenue growth strategies and margin management across the consumer discretionary, financials, and information technology sectors?",
    "role": "Equity Research Analyst",
    "type": "multi_sector_single_time",
    "_hf_config": "questions",
    "_hf_split": "train",
    "_question_file": "global_questions_new",
    "_row_id": "global_questions_new_1"
  }
]
```

### corpus / train

- Reused existing local file: `exp_ect_qa\processed\raw_corpus_train.csv`
