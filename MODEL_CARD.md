# Model / system card

## Intended use

Measure how chunk size and overlap affect retrieval quality and context volume. Educational local experiment; no production deployment is implied.

## Data

Synthetic arrays, manually written documents or replayed JSON. No real user records. Seeds and sample inputs are recorded in app.py and sample_input.json.

## Method

Generate fixed-word windows while retaining source identity and offsets. For each configuration rank chunks with BM25, deduplicate sources for recall, and compute reciprocal rank and returned word counts.

## Evaluation protocol

Evaluate chunk sizes 12, 24 and 48 with zero or 25 percent overlap on the same labeled development queries. Save per-query evidence and aggregate recall@3 and MRR.

## Known limits

This is a retrieval experiment, not a claim about generated-answer quality. Configurations compared on development queries need a fresh final test set. Word windows can split sentences.

## Verification status

Local tests and default experiment are executed during packaging. Live model generation is not exercised in the build. A successful validation result is limited to the checks implemented in code.

## Further work

Add sentence-aware chunking, token budgets, query stratification, answer support labels and independent held-out retrieval evaluation.
