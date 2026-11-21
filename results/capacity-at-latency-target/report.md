<!-- register: technical benchmark report | reader: applied AI researchers and serving engineers checking the evidence | consumed: methods and results review after the README -->
# Two TP4 replicas with DSpark K5 met a 30-second p95 at up to about 0.54 eight-turn conversations/s per node, with thinking on

On one eight-B200 node, two four-GPU tensor-parallel replicas (`2xtp4`) with static DSpark speculative decoding (K5) kept completion p95 at or under 30 s on every run up to 0.544 W1 or 0.267 W2 conversations per second per node, the highest tested loads where all runs passed (one arrival draw each). At 5 to 6% more W1 load, one of three draws passed. Without speculation, the same engine configuration on the other four GPUs missed the target at the lowest load tested in both workloads. One eight-way data- and expert-parallel engine (`dep8`) with K5 had 1.5 to 2.4 times `2xtp4`'s completion p95 at the same loads (median of runs on the same draw). Measurements ran on November 19 and 20, 2026.

All loads in this report are **measured arrivals**: conversations whose first turn was scheduled in the 360-second measured period of the realized arrival schedule, divided by 360. In every window that did not stop early, all of those first turns were dispatched. The load generator's rate setting is kept only as an identifier in the tables, because single Poisson draws over 360 s landed up to 18% from it.

## Setup

| Item | Value |
| --- | --- |
| Model | DeepSeek-V4.1-Flash-1017, revision `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Image | `vllm/vllm-openai@sha256:aeb580d55ee3dcccd7b5c78d7871326f367bee5cfd8ca8e411c253d7983a0046` (vLLM build `e0d00e4b8`), the same image as the other studies in this repository. DSpark, `--engram-config` and the DSV41 attention backends are used as shipped in this nightly image |
| `2xtp4` K5 | Two engines, tensor parallel 4, GPUs 0-3 and 4-7; `FLASHINFER_MLA_SPARSE_DSV41` attention, FP8 KV cache, prefix caching, Engram CPU offload, 64 sequences per engine; DSpark with five draft tokens, probabilistic draft sampling, block rejection, adaptive verification off; 40,819,149 KV tokens per engine (engine startup log) |
| `2xtp4` NoSpec | The same arguments without speculation, on GPUs 4-7, with `--num-gpu-blocks-override 399442` so its KV capacity equals the K5 engine's 40,819,149 tokens (it would otherwise allocate 437,455 blocks) |
| `dep8` K5 | One engine, data parallel 8 with expert parallelism; `FLASHMLA_MEGA_ATTN_DSV41` attention, `fp8_ds_mla` KV cache, `deep_gemm_mega_moe` MoE backend with FlashInfer autotuning, Engram CPU offload with embeddings shared across data-parallel ranks, 128 sequences per rank, the same K5 settings; it omits `--enable-scale-out` and per-request speculative metrics |

[configuration.json](configuration.json) has every serving argument, CPU placement, the pass rules and the producer code hashes.

**Workloads.** W1 has eight sequential turns with prompts from 48,697 to 116,205 tokens (median 51,762 at turn 1 rising to 73,428 at turn 8). W2 has four turns from 243,476 to 445,066 tokens (median 249,480 to 259,513). Prompts are public Python source and public-domain prose ([input-origin.json](input-origin.json)). Earlier assistant turns come from a frozen transcript, so generated answers do not change later prompts. Every request has thinking enabled (reasoning effort 50), temperature and top-p 1.0, a 4,096-token output cap, 12 tool definitions and its own seed. Paired arms receive identical request bodies and seeds; their outputs differ because sampling is random.

**Load.** Each window has 60 s of warm-up arrivals, 360 s of measured arrivals and up to 120 s to drain. The first turn of each conversation arrives on a Poisson schedule; later turns follow back-to-back as each one completes. A window stops early after 120 s with a continuously nonempty waiting queue, or at once if more than 256 requests are waiting. Each window uses its own cache namespace, so prefix caching helps within a conversation and not across windows.

**Pass rule.** Completion p95 at or under 30 s and first-output p95 at or under 10 s, over every scheduled measured turn, with all work finished and the engine queue empty by the end of the 120 s drain; a turn that does not finish counts as infinitely late, and p95 is undefined (n/a) when more than 5% of turns did not finish. For `2xtp4`, each replica receives its own copy of the stream and both replicas must pass; the node p95 shown pools them. For `dep8`, two streams share the engine and the verdict pools them; a single stream's drain check can fail while the other stream is still running, so only the pooled verdict counts. The two rules differ: pooled, `2xtp4` would also pass W1 at 0.572 (29.7 s) and one of three W2 draws at 0.300 (29.9 s); per stream, `dep8` would pass no W1 run at 0.350. Turns that hit the output cap or a repetition heuristic count as finished; their rates are reported below.

## Capacity of `2xtp4` K5

A draw is one Poisson arrival schedule; a run is one execution of a draw. Input tokens/s is the offered prompt load: prepared prompt tokens of all scheduled measured turns over 360 s, including the cached prefix. Where every turn completed, it matches the server-reported prompt tokens within 0.0005%.

### W1

| Measured arrivals per node (/s) | Input tokens/s per node | Draws × repeats | Passed | Node completion p95 (s) |
| ---: | ---: | --- | ---: | --- |
| 0.350 | 183K | 1 draw | 1 of 1 | 19.8 |
| 0.533 | 276K | 1 draw | 1 of 1 | 27.3 |
| 0.539 | 277K/284K | 2 draws, 4 runs | 4 of 4 | 26.6, 27.5, 28.2, 28.9 |
| 0.544 | 280K | 1 draw | 1 of 1 | 26.7 |
| 0.572 | 295K | 1 draw | 0 of 1 | 29.7 |
| 0.578 | 297K/299K | 2 draws | 1 of 2 | 30.5, 27.8 |
| 1.061 | 560K | 1 draw, 3 runs | 0 of 3 | n/a, n/a, n/a |

### W2

| Measured arrivals per node (/s) | Input tokens/s per node | Draws × repeats | Passed | Node completion p95 (s) |
| ---: | ---: | --- | ---: | --- |
| 0.161 | 170K | 1 draw | 1 of 1 | 18.4 |
| 0.211 | 219K | 1 draw, 3 runs | 3 of 3 | 27.1, 22.6, 24.8 |
| 0.222 | 233K | 1 draw | 1 of 1 | 21.4 |
| 0.256 | 267K | 1 draw | 1 of 1 | 26.7 |
| 0.267 | 278K | 1 draw | 1 of 1 | 28.7 |
| 0.272 | 284K | 2 draws | 1 of 2 | 27.8, 30.9 |
| 0.300 | 310K/312K | 3 draws | 0 of 3 | 40.5, 35.6, 29.9 |
| 0.372 | 387K | 1 draw, 3 runs | 0 of 3 | 44.7, 44.9, 45.4 |

The W1 node value of 29.7 s and the W2 value of 29.9 s failed because one replica exceeded 30 s. First-output p95 in passing node runs was 0.80 to 0.92 s on W1 and 4.07 to 6.19 s on W2.

The 284K draw at 0.539 also ran three more times on replica 0 alone, beside the overloaded NoSpec engine on the other GPUs ([K5 against NoSpec](#k5-against-nospec)); one of those runs reached 30.3 s. Counting every K5 replica run of that draw, 8 of 9 met the target.

**The limit tracks total prompt tokens, not uncached prefill.** Within `2xtp4` K5, all 9 draws below 140K input tokens/s per replica passed, and 3 of the 9 draws from 141.9K to 156.1K passed (one of them in three runs). Both workloads stop passing in the same band even though W2 conversations are about twice as long and arrive half as often. The prefix cache served 85.5% of W1 prompt tokens and 74.6% of W2, so at the limit the uncached prefill was about 20.7K tokens/s per replica on W1 and 36.0K on W2. The workloads match on total prompt tokens including cached prefix, not on prefill work; no measurement here identifies the mechanism.

**Prompt totals differ slightly between draws.** Prompts are calibrated to per-turn token targets, so totals per conversation are nearly constant within one prepared population (W1: 527,318.6 and 527,318.3 tokens for two schedules over the same population) and differ by up to 2.5% across populations (W1: 514K to 527K). That is why the input-token column can differ at equal arrivals.

**One draw is not enough near the limit.** The two replicas receive identical streams, so their difference measures engine-level variation only. At 0.578 W1 conversations/s per node, two draws gave node p95 of 27.8 and 30.5 s, while the largest difference between the two replicas in any W1 run was 1.9 s and three runs of one draw at 0.539 spanned 1.4 s. At 0.300 W2 conversations/s per node, three draws spanned 29.9 to 40.5 s, while the largest replica difference at that load was 5.6 s; three runs of one W2 draw at 0.211 spanned 4.5 s. The W1 comparison rests on two draws. Most levels near the limit have one draw: the W1 limit at 0.544, and the W2 levels at 0.256 and 0.267.

## K5 against NoSpec

The K5 engine on GPUs 0-3 and the NoSpec engine on GPUs 4-7 ran at the same time with identical streams and shared host CPUs. Values are per engine, three runs of one draw.

| Workload | Measured arrivals per replica (/s) | K5 completion p95 (s) | NoSpec completion p95 (s) | K5 first-output p95 (s) | NoSpec first-output p95 (s) |
| --- | ---: | --- | --- | --- | --- |
| W1 | 0.175 | 20.8, 20.6, 19.2 | 53.8, 48.9, 55.7 | 0.78, 0.80, 0.79 | 1.31, 1.35, 1.27 |
| W1 | 0.269 | 30.3, 29.1, 27.1 | n/a, n/a, n/a | 0.88, 0.89, 0.87 | n/a, n/a, n/a |
| W1 | 0.531 | n/a, n/a, n/a | n/a, n/a, n/a | n/a, n/a, n/a | n/a, n/a, n/a |
| W2 | 0.081 | 18.3, 18.4, 16.9 | 55.0, 54.7, 55.1 | 4.07, 4.16, 4.21 | 8.34, 7.24, 8.80 |
| W2 | 0.106 | 26.4, 28.1, 25.5 | 84.0, 93.1, 86.1 | 5.30, 6.35, 5.53 | 12.46, 12.54, 12.68 |
| W2 | 0.186 | 45.1, 47.9, 50.9 | n/a, n/a, n/a | 6.75, 7.49, 7.72 | n/a, n/a, n/a |

NoSpec missed the 30 s target at the lowest load in both workloads, so its own limit is below 0.175 (W1) and 0.081 (W2) conversations/s per engine and was not located. NoSpec also left 6 of 1,512 W1 turns unfinished at 0.175.

A separate paired capture on November 19 ran the same two engine configurations side by side on identical inputs ([k5-nospec-paired.csv](tables/k5-nospec-paired.csv)): W1 at 0.175 conversations/s per engine over a 360 s window and W2 at 0.077 over a 600 s window, one draw each. NoSpec was beyond its limit at both loads (completion p95 52.8 and 45.5 s, against 19.7 and 16.9 s for K5), so the ratios below include NoSpec's queueing as well as its decoding. Over 502 matched W1 turns and 184 W2 turns, the geometric mean of K5 completion time over NoSpec completion time was 0.376 on W1 and 0.469 on W2, and per-turn decode rate was 2.46 and 2.62 times higher. The 95% bootstrap intervals (0.349 to 0.404 and 0.402 to 0.549 for completion time; 2.39 to 2.53 and 2.38 to 2.91 for decode rate) resample conversations within the one draw. Output lengths differ between the arms and are not in the table. Within 30 s, K5 finished 503 of 504 W1 turns against 392, and 184 of 184 W2 turns against 150.

## `dep8` K5

| Workload | Measured arrivals per node (/s) | Passed | `dep8` node completion p95 (s) | `2xtp4` K5 on the same draw (s) |
| --- | ---: | ---: | --- | --- |
| W1 | 0.350 | 1 of 3 | 30.3, 30.1, 29.5 | 19.8 |
| W1 | 0.539 | 0 of 3 | 58.0, 60.7, 59.9 | 27.5, 28.2, 28.9 |
| W1 | 1.061 | 0 of 3 | n/a, n/a, n/a | n/a |
| W2 | 0.161 | 0 of 3 | 47.0, 40.5, 43.2 | 18.4 |
| W2 | 0.211 | 0 of 3 | 57.5, 51.2, 72.1 | 27.1, 22.6, 24.8 |
| W2 | 0.372 | 0 of 3 | n/a, 116.2, n/a | 44.7, 44.9, 45.4 |

No `dep8` load passed all three runs. Taking the median of the runs at each load on the same draw, `dep8` completion p95 was 1.5 times `2xtp4`'s on W1 at 0.350, 2.1 times at 0.539, and 2.3 to 2.4 times on W2 at 0.161 and 0.211; at W2 0.372 two `dep8` runs have no defined p95. The `dep8` and `2xtp4` runs were made five to six and a half hours apart on the same node.

The configurations differ in more than GPU grouping: attention backend, KV format, MoE backend, sequence limits and prefix-cache locality. Each `2xtp4` stream stays on one replica, while `dep8` places requests across its data-parallel ranks without session affinity. At the W1 loads compared above, the prefix cache served 59% to 68% of prompt tokens on `dep8` against 85.4% on `2xtp4`; on W2 it served 52% to 58% against 74.6%. The earlier [multi-turn load study](../sustained-speculation-layouts/report.md), which routed each conversation to one rank, measured a much smaller `dep8` gap. These results compare two complete configurations and do not isolate data or expert parallelism.

## Speculative acceptance and a trace-replay control

Acceptance length is `1 + accepted draft tokens / draft steps`, from vLLM's speculative-decoding counters over each window; it counts the token every step produces. On real-text prompts it was 2.90 to 3.00 on W1 and 3.05 to 3.20 on W2 for `2xtp4`, and 2.93 to 2.97 on W1 and 3.12 to 3.18 on W2 for `dep8` ([arms.csv](tables/arms.csv)). Draft-token acceptance on W1 `2xtp4` ranged from 38.0% to 40.0%.

A separate control replayed a private 15-minute request trace (arrival times, prompt and output lengths, prefix-hash IDs; the trace is not published) with `vllm bench serve` and its `timed_trace` dataset, against replica 0 only. That dataset fills each prefix-hash chunk with token IDs sampled uniformly from the vocabulary (`TimedTrace._sample_token` in `vllm/benchmarks/datasets/datasets.py` at build `e0d00e4b8`). The replay also used the plain completions endpoint without a chat template, thinking or tools, and forced output lengths to the trace's values. Mean acceptance length was 1.81 (0.81 accepted draft tokens per step; 16.3% of draft tokens accepted), and 1.80 in a second replay at twice the request rate. On the same engine, a W1 window on the 284K draw gave 2.96 (1.96 accepted per step; 39.2%). The replay's completion p95 was 56.9 s. No NoSpec replay was run, and the replay differs from W1 in prompt content, endpoint, output control and load, so the gap is not attributed to random tokens alone.

Random-token inputs are a known hazard for speculative-decoding benchmarks. [SPEED-Bench](https://arxiv.org/abs/2604.09557) reports two failure modes: trivial responses that raise acceptance and topic latching that lowers it; in its tests random tokens overstated speculative throughput by about 23% on average and also skewed MoE expert routing. In this replay acceptance fell. Trace replays used to size a speculative decoder need real-text prompts.

## Turn quality flags

At W1 0.175 per replica, over all 1,512 scheduled turns per arm (NoSpec finished 1,506), 76 K5 and 73 NoSpec turns hit the 4,096-token cap, and 51 and 66 were flagged by the repetition heuristic. On W2 at 0.081 and 0.106 combined, where both arms finished all 804 turns, 38 and 32 hit the cap and 35 and 26 were flagged. The November 19 paired capture gave 21 and 31 capped and 20 and 18 flagged on W1, and 7 and 10 capped and 11 and 5 flagged on W2. Higher loads are left out because NoSpec sent fewer turns there. These flags are heuristics on single samples at temperature 1.0, not a task-quality measurement, and they show no consistent difference between the arms.

## Limits

- One node and one model revision; no fresh-node replication.
- Two synthetic real-text workloads with fixed transcripts. Generated answers are not scored, so these results say nothing about correct-task throughput.
- Most loads near the limit rest on one arrival draw; the W1 limit at 0.544 and the W2 limit at 0.267 each rest on one.
- `2xtp4` and `dep8` verdicts use different pooling rules, stated above.
- NoSpec ran on one engine only, and its limit was not found.
- `dep8` used one engine shared by two streams; its counters cannot be split by stream.
- The mechanism behind the input-token match is not measured.
- The [multi-turn load study](../sustained-speculation-layouts/report.md) in this repository used forced 400-token outputs with thinking off and a session-affinity router; its W1 completion p95 of 4.17 s at 0.531 measured conversations/s per node is not directly comparable. Output length is the likely main difference, but prompts and client also differ.

## Files

| File | Contents |
| --- | --- |
| [tables/node-cells.csv](tables/node-cells.csv) | Node-level verdicts for `2xtp4` K5 and `dep8` K5, with `dep8` pooled first-output p95 and drain status |
| [tables/arms.csv](tables/arms.csv) | Per replica or stream: measured arrivals, offered input tokens/s, latency quantiles, drain status, acceptance, prefix-cache hit fraction |
| [tables/k5-nospec-paired.csv](tables/k5-nospec-paired.csv) | November 19 paired turn ratios with bootstrap intervals |
| [configuration.json](configuration.json) | Serving arguments, placement, request settings, pass rules, workload lineage, trace-replay arguments, producer code hashes |
| [input-origin.json](input-origin.json) | Prompt text origins, hashes and licenses |

Raw request records, engine telemetry and logs are retained privately with per-file SHA-256 hashes.
