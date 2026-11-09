<!-- register: repository guide | reader: benchmark reproducer | consumed: setup and evidence check -->
# DeepSeek V4.1 Flash 1017 on eight B200 GPUs

The [finalist validation study](results/finalist-validation/README.md) compares NoSpec and static DSpark K5 on two TP4 replicas. At six offered requests/s, K5 completes about 9% more correct answers within one second in two short repeated-question windows; unrestricted correct-answer rate changes by less than 1%. Concurrent natural-output token throughput also favors K5 on the observed block medians. Shared engine/cache history, a selected quality panel and incomplete queue-drain sampling limit these findings.

The completed [fixed-configuration comparison](results/fixed-comparison/README.md) contains 336 accepted cells, 1,008 repetitions and 29,508 measured requests. The separate [1M-input supplement](results/context-1m-output2048/report.md) contains 44 accepted cells, 132 repetitions and 3,552 measured requests with 2,048 output tokens. The original 1M-input panel uses 512 output tokens; keep those panels separate.

The [charts](results/fixed-comparison/report/README.md) and sanitized tables describe finite bursts across context length and concurrent demand. They include repetition spread and achieved cache reuse. Backend differences, cache conditions and temporal drift limit comparisons. Sustained capacity, broad task quality and production reliability require separate validation.

The completed [tuning screens](results/tuning-adaptation/README.md) add 12 matched comparisons, 72 measured repetitions and 324 measured requests. Static DSpark K5 is inconclusive under the declared latency guardrail; the tested graph-capture and prefill-budget candidates were rejected under their declared criteria. That historical decision remains unchanged. The later finalist study above reports different workloads and metrics separately.

The [comparison contract](docs/comparison-contract.md) defines the workload and timing boundaries. The [configuration guide](docs/frozen-configurations.md) records runtime and model pins, GPU grouping, backend choices and CPU placement.

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

The checked-in harness supports the local fixtures and public result checks below. Live node execution requires source-level operator assembly and retains campaign-specific preconditions; this is not a turnkey fresh-node deployment. The [reproduction notes](docs/reproduction-notes.md) identify those requirements. Historical measurements can use different immutable producer versions. Result provenance must identify those original versions; running a newer verifier does not replace an original-source audit.

[Reproduction notes](docs/reproduction-notes.md) describe ownership, evidence and timing precautions. The [publication policy](docs/publication-policy.md) keeps raw captures and operational failure artifacts outside Git while preserving material scientific outcomes.

## Verify the completed result exports

```sh
python3 -B reproduce/verify_result_package.py results/fixed-comparison
python3 -B reproduce/verify_result_package.py results/context-1m-output2048
python3 -B reproduce/verify_result_package.py results/tuning-adaptation
python3 -B reproduce/verify_result_package.py results/finalist-validation
```

These checks validate public file integrity and available table arithmetic. Raw responses, failure logs, interrupted attempts and private infrastructure records remain in separate evidence storage. Public provenance identifies original source versions by their content hashes.
