<!-- register: repository policy | reader: contributor | consumed: preparing a commit -->
# Publish reproducible code and sanitized results

The repository contains source, dependency pins, tests, scientific configurations and reviewed result tables. Runtime captures, raw prompts and responses, logs, tracebacks, attempt journals, incident histories, connection details and credentials belong in separate private evidence storage.

Rejected and inconclusive experiment outcomes remain in the published results when they affect interpretation. A rejected optimization is a scientific finding. Its operational failure logs remain private. Tests that exercise error handling are source code and belong with the harness.

Use explicit paths when staging files. Before each commit, inspect `git diff --cached --stat` and run:

```sh
python3 -B reproduce/verify_publication.py --staged
git diff --cached --check
```

The publication verifier reads the staged bytes, checks the allowed paths and rejects common credential forms, private connection details and non-English publication text. Results must appear in the explicit `configs/publication-results.json` inventory. Each PNG must also match its reviewed SHA-256, binding its pixels, metadata and all other bytes. Adding a result or changing a chart requires an explicit inventory update and review.

The verifier uses the inventory from the Git index, so an unstaged approval cannot authorize a staged file. Every indexed result is checked even when the change only updates the inventory. The verifier never prints matched credential values. This check complements review; it cannot establish that every possible sensitive value has been removed. Git ignore rules do not protect files already staged or tracked.

Raw evidence remains necessary for scientific acceptance and is retained privately. Public checksums identify the selected source and result bytes. Public table verification establishes arithmetic and coverage; replaying original raw-stream and lifecycle audits requires the separately retained original evidence and its matching producer tools.
