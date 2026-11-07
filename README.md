<!-- register: repository guide | reader: benchmark reproducer | consumed: setup and evidence check -->
# DeepSeek V4.1 Flash 1017 on eight B200 GPUs

This harness compares four fixed serving configurations across context length and concurrent demand. The [comparison contract](docs/comparison-contract.md) defines the 336-cell workload, timing boundaries and claim limits. The [configuration guide](docs/frozen-configurations.md) records runtime and model pins, GPU grouping, backend choices and CPU placement.

Requires Python 3.12 or later for the harness. Local fixtures bind ephemeral loopback ports. Use fresh output directories outside the checkout:

```sh
python3 -B reproduce/campaign.py plan --plan docs/comparison-contract.md --output /tmp/dsv41-ledger.json
python3 -B reproduce/campaign.py fixture --output /tmp/dsv41-fixture
python3 -B reproduce/verify_fixture.py /tmp/dsv41-fixture
python3 -B reproduce/campaign.py fixture-burst --output /tmp/dsv41-fixture-burst
python3 -B -m unittest discover -s tests -v
```

Fixtures exercise deterministic streaming, persistence, interruption and recovery with synthetic clocks. They establish local behavior within their assertions; live GPU qualification and independently audited measurements have separate evidence requirements.

`reproduce/launcher.py` renders digest-pinned Docker and router arguments with disjoint GPU groups, loopback ports and inference retries disabled. The supervised path binds tokenizer inputs, explicit profiles, semantic readiness, safety monitoring, worker evidence, excluded warmup, request capture and exact-container cleanup. Rendering alone starts no processes.

The checked-in harness is a portable source snapshot. Historical measurements can use different immutable producer versions. Result provenance must identify those original versions; running a newer verifier does not replace an original-source audit.

[Reproduction notes](docs/reproduction-notes.md) describe ownership, evidence and timing precautions. The [publication policy](docs/publication-policy.md) keeps raw captures and operational failure artifacts outside Git while preserving material scientific outcomes.
