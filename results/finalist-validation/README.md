<!-- register: benchmark result guide | reader: researcher and inference engineer | consumed: compare scoped findings and reproduce calculations -->
# Static K5 improves concurrent token throughput and deadline-qualified correct-answer rate

The [report](report.md) compares two TP4 replicas using NoSpec and static DSpark K5 on one eight-B200 node. Concurrent natural-output token throughput favors K5 on the observed block medians, with wide variation. At six offered requests/s, K5 completes about 9% more correct answers within one second in two short mixed-load windows; unrestricted correct-answer rate changes by less than 1%.

The [reproduction guide](REPRODUCE.md), [data dictionary](DATA.md), prompt-free observations and four PNG/SVG charts preserve metric definitions and limitations. The [earlier tuning study](../tuning-adaptation/README.md) remains inconclusive under its different predeclared latency guardrail. The newer workload-specific observations do not reverse that historical decision.

Queue-drain characterization remains incomplete. Shared cache/engine history, repeated questions and the selected quality panel limit inference. General quality equivalence, a maximum sustainable service rate, fresh-node replication and production readiness remain unestablished.

From this directory:

```sh
python3 -B reproduce/replay_tables.py
```

From the repository root:

```sh
python3 -B reproduce/verify_result_package.py results/finalist-validation
```

These checks reproduce table arithmetic and file integrity, rather than executing inference or independently rescoring model answers. Full raw evidence remains privately retained.
