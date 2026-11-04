# Technical revision 1

Offline audit only. Historical manifests, raw answers, Florence regions and human ratings are unchanged.

`summary.json` separates received API responses, full schema-valid responses, API errors, schema/finish errors and skips. `responses.csv` retains per-call diagnostics and context-budget checks. `source-sha256.json` records source bytes. Reused prompt40 baseline responses are not additional generations.

Legacy image identity is verified through the frozen input SHA and original image bytes. Model digests are compared with historical records; installed weights were not queried. Historical pre-call token budgets cannot be reconstructed as exact tokenizer counts. Post-call counts are reported without claiming that they prove no server-side truncation. No content accuracy is evaluated.
