<!-- register: tuning report | reader: serving engineer | consumed: measured tradeoffs and limitations -->
# DSpark, graph capture and prefill-budget screens

Three serial screens were planned: static DSpark K5, one graph-capture configuration, and 8192 versus 16384 batched-token budget. Each complete screen uses four matched workloads and three paired repetitions per arm.

DSpark is the tested speculative decoding method. K5 sets five speculative tokens per draft step; this candidate uses probabilistic draft sampling with adaptive verification disabled. Its [recorded profile](profiles/dspark-candidate.json) defines the complete candidate configuration.

| Screen | Outcome | Audited measured requests | Reserved total requests | Node hours |
| --- | --- | ---: | ---: | ---: |
| dspark | inconclusive | 108 | 146 | 3.35 |
| graph | rejected | 108 | 166 | 6.02 |
| prefill | rejected | 108 | 140 | 1.62 |

The bounded live DSpark qualification observed 18 draft steps, 90 proposed draft tokens and 52 accepted draft tokens across 6 probes. This establishes real rejection in that qualification, not a general quality or efficiency claim.

Decode TPOT is the elapsed interval from first to last visible output event divided by the completion-token count minus one. TTFT denotes first visible output latency in these screens. End-to-end latency runs from request dispatch to terminal completion; throughput divides verified output tokens by the measured burst makespan. Each arm's summary is the median of its three repetition summaries.

## Static DSpark K5 on two TP4 replicas

The DSpark screen was **inconclusive** under a primary metric of decode-window time per output token.

| Input tokens | Concurrency | Decode TPOT ratio | TTFT ratio | End-to-end ratio | Throughput ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8000 | 1 | 0.563 | 0.619 | 0.577 | 1.732 |
| 8000 | 8 | 0.585 | 0.222 | 0.505 | 1.834 |
| 64000 | 1 | 0.493 | 1.075 | 0.547 | 1.829 |
| 64000 | 8 | 0.715 | 0.715 | 0.715 | 1.342 |

Each ratio is candidate divided by baseline; lower is better for latency, higher for throughput. The recorded decision uses the predeclared primary and guardrail rules, including paired-repetition and noise checks.

At 64K input and concurrency one, the median first-output latency ratio was about 1.075. Its three paired ratios straddled the 1.05 guardrail. That noisy secondary result kept the screen inconclusive despite the lower decode-window time.

## Graph capture on two TP4 replicas

The graph screen was **rejected** under a primary metric of aggregate burst throughput.

| Input tokens | Concurrency | Decode TPOT ratio | TTFT ratio | End-to-end ratio | Throughput ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8000 | 1 | 0.984 | 1.471 | 1.094 | 0.914 |
| 8000 | 8 | 1.125 | 1.289 | 1.098 | 0.889 |
| 64000 | 1 | 0.986 | 1.627 | 1.047 | 0.955 |
| 64000 | 8 | 0.956 | 1.167 | 0.997 | 1.003 |

Each ratio is candidate divided by baseline; lower is better for latency, higher for throughput. The recorded decision uses the predeclared primary and guardrail rules, including paired-repetition and noise checks.
Native queried and padded shapes, dispatch decisions and fallback modes, and reported graph capture memory were checked in every retained measured parent. These are pre-execution decisions, not proof of GPU execution; interval aggregates can include readiness and qualification traffic and are not attributed to individual requests.
Two graph passes required postcapture evidence recovery: one protected hard-link projection and one interrupted observer snapshot. Their original failures, captured bytes and attempt histories were privately retained, independently compared with reconstructed evidence, and strictly verified without replaying requests.

At concurrency eight, throughput ratios were about 0.889 at 8K input and 1.003 at 64K. The declared primary rule required at least 1.10 at both lengths, with positive paired repetitions and separated repetition ranges. This rejects this capture-size configuration under these rules; it does not reject graph capture generally.

## Prefill budget on DEP8

The prefill screen was **rejected** under a primary metric of first visible output latency.

| Input tokens | Concurrency | Decode TPOT ratio | TTFT ratio | End-to-end ratio | Throughput ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128000 | 1 | 1.003 | 0.940 | 0.989 | 1.011 |
| 128000 | 8 | 1.000 | 1.006 | 1.001 | 1.001 |
| 256000 | 1 | 1.002 | 0.784 | 0.956 | 1.046 |
| 256000 | 8 | 0.974 | 0.805 | 0.937 | 1.066 |

Each ratio is candidate divided by baseline; lower is better for latency, higher for throughput. The recorded decision uses the predeclared primary and guardrail rules, including paired-repetition and noise checks.

At 128K input and concurrency one, first-output latency improved by about 6.0%, below the declared 10% requirement. The roughly 21.6% improvement at 256K did not satisfy the rule requiring both lengths to pass.

## Scope and limits

- The primary threshold was 10% improvement; secondary guardrails were 5%, using three matched repetitions rather than treating requests as independent repetitions.
- Visible stream events can contain multiple tokens, so the decode timing measure is a stream-window proxy rather than exact token interarrival time.
- Graph measurements include the same native metrics logging and worker observer in both arms; their overhead is part of the measured configuration. Finite screen results do not establish production adoption, capacity, or general quality equivalence.
- Request limits are conservative reservations and can exceed the number of completed measurements. Raw evidence, original and reconciled attempts, source pins and private version-readback receipts are retained separately.

The [comparison records](comparisons.json) retain every per-repetition metric and the [contracts](contracts/) retain the decision rules. [Profile records](profiles/) contain normalized templates and exact model/image identities; placeholder paths require an operator-supplied deployment configuration. The broader performance search is still in progress and is excluded here.
