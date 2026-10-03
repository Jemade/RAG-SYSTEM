# Evaluation data

The 12 corpus files and 60 labeled questions are an authored fictional operations example. No external customer records are included. The references were specified separately from the retrieval/generation code.

Each JSONL row has an ID, question, accepted answer list, relevant document sources, failure category, scoring kind (`exact` or `open`), rubric and split. Forty factual/abstention questions use exact match; twenty explanatory questions require a judge. There are 45 development and 15 holdout rows.

The initial cases probe expected weaknesses. `rag-system triage` exports failures observed in an actual evaluation for human review and additional regression authoring. Never substitute a model's own answer for a reviewed reference. Source relevance labels are document-level, not exact chunk-level. No-answer rows have no relevant source and retrieval metrics are intentionally unscored.
