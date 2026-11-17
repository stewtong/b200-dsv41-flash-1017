<!-- register: aggregate reproduction note | reader: serving engineers | consumed: installed repository verification -->
# Verify aggregate provenance and portable guards

From the integrated repository root:

```sh
python3 -B reproduce/verify_result_package.py results/sustained-speculation-layouts
```

The verifier checks final inventory, 39 core rows and the cache-confounded diagnostic, accepted audit joins, coverage, source hashes and SVG sources. It verifies stored offline regression receipts. It does not replay captures, repeat raw audits or reproduce inference.

Run the five portable-helper cases against this checkout, with local streams and tokenizer responses patched in memory:

```sh
python3 -B reproduce/check_sustained_portable.py
```

The expected result is five passed cases with no network or inference. Temporary fixture receipts are written outside the result package. The aggregate verifier never imports Python from a supplied result directory.

Check both figures against their published input values without rewriting them:

```sh
python3 -B reproduce/render_sustained_speculation.py results/sustained-speculation-layouts --verify
```

`capture-source/` preserves historical measured source variants under original hashes. Exact raw streams, request bodies and corpus text remain private; `input-origin.json` carries public provenance and licenses. `audit-summary.json` records accepted old and new audit scope.

`portable-candidate/` retains private-legacy-cache and strict-EOS guards and adds the optional-None forced-smoke arithmetic guard. `portable-fixtures.json` records five offline helper cases with patched local streams/tokenizer: complete-stop acceptance, truncated and length rejection, the optional-None case and an explicit bad receipt. This is no fresh-node or full-benchmark reproduction and changes no scientific values.
