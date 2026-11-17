# Bounded portable guards

Historical producer sources under `capture-source/` remain byte-identical to their measured hashes. `portable-candidate/` is separate and does not change their scientific meaning.

The candidate retains the optional private-legacy-cache guard and strict natural-stop requirement. A third guard skips the optional numeric `base_tokens` comparison when that argument is None, after saving the forced-smoke receipt. A supplied value still enforces the comparison; an invalid receipt always fails.

Offline fixtures exercise strict EOS acceptance and rejection plus the optional-None and invalid-receipt cases. They do not run inference, establish fresh-node reproduction or alter benchmark results.
