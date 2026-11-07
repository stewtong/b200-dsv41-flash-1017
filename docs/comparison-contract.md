<!-- register: benchmark contract | reader: benchmark reproducer | consumed: workload and claim verification -->
# Compare four fixed configurations across context length and demand

This experiment compares DeepSeek V4.1 Flash 1017 on eight B200 GPUs using four fixed NoSpec deployments. It measures finite request bursts. It makes no claim of globally optimal tuning, pure topology causation, sustainable serving capacity, task-quality equivalence or production reliability.

| Configuration | Allocation across eight GPUs | Attention and expert parallelism |
| --- | --- | --- |
| `2xtp4` | Two four-GPU replicas | Attention TP4 per replica |
| `4xdep2` | Four two-GPU replicas | Attention DP2/TP1 and EP2 per replica |
| `2xdep4` | Two four-GPU replicas | Attention DP4/TP1 and EP4 per replica |
| `dep8` | One eight-GPU replica | Attention DP8/TP1 and EP8 |

Freeze the exact launcher profile, image, checkpoint, tokenizer, routing, admission, graph, attention/expert backend, KV dtype and Engram placement for each configuration. Necessary differences remain visible. Do not change a profile silently for a larger context or higher concurrency. Existing qualified NoSpec settings are the comparison configurations, without an optimization tournament.

## The matrix has 336 cells and three repetitions per cell

| Axis | Cold ladder | Warm-prefix panel |
| --- | --- | --- |
| Configurations | All four | All four |
| Rendered input tokens | 8,000; 16,000; 32,000; 64,000; 128,000; 256,000; 400,000; 512,000; 1,000,000 | 128,000; 256,000; 512,000; 1,000,000 |
| Node-wide outstanding requests | 1, 2, 4, 8, 16, 32, 64, 128 | 1, 8, 32 |
| Cells | 288 | 48 |
| Repetitions | 864 | 144 |
| Measured requests | 27,540 | 1,968 |

Each repetition sends a finite burst of the declared number of requests and waits for terminal outcomes. It does not replenish completed requests. Outstanding demand includes queued work; it is not a count of simultaneously decoding sequences.

Use 2,048 output tokens below 1M input and 512 output tokens at 1M. Keep these different decode budgets visible in the results. Use `fixed-performance-v1`: streamed Chat Completions with final usage, thinking disabled, temperature 1.0, top-p 1.0, `min_tokens=max_tokens` and `ignore_eos=true`. Preserve exact prompt bytes, tokenizer/encoder identity and matched seeds across configurations. Confirm rendered input and returned usage within 0.25%, exact requested output, valid UTF-8 and nonempty content. A leading identifier, ASCII-only prose or a task-quality score is not a performance acceptance requirement.

## Keep warmup and cache conditions visible

Warm kernels with explicitly excluded prompts whose early namespaces are disjoint from scored prompts. Record the actual warmup shape; one warmup shape does not establish that every later graph or context shape is warmed. Keep startup and warmup outside scored timing.

Cold requests use unique early prefixes and available supported cache reset/state evidence. When flush/state proof is unavailable, label them **cache-busted cold** and disclose that limitation. Retain measured cache counters.

For a warm-prefix request, condition a prefix of target length minus 256 tokens, rounded down to the actual cache block boundary. Append the remainder to reach the target. Condition independently per repetition with a stable session map. A warm-designated miss or eviction remains in the warm-workload result with its achieved hit fraction and latency; never relabel or drop it.

Use the same front door, no inference retries, matched prompt bodies and explicit session/rank placement. Rotate low-concurrency placement over repetitions and record balancing limitations. Retain execution order and repetition spread. Unresolved variation prevents an unsupported winner; it does not erase a valid descriptive observation.

## Derive metrics from retained raw records

- Burst aggregate output rate: valid completed output tokens divided by first-dispatch-to-last-terminal makespan. Failed and timed-out requests remain in elapsed time and request denominators; emitted partial tokens are separate.
- First output: the first nonempty output content chunk. Initial metadata does not count. Report first visible text separately when whitespace precedes it.
- Decode-window rate: `(completion_tokens - 1) / (last_output_time - first_output_time)` when defined. Multi-token chunks make this a stream-derived estimate, not precise per-token timing.
- Completion latency: terminal time minus dispatch time. End-to-end output rate divides valid completion tokens by that interval and is distinct from decode-window rate.
- Report completion/error counts, requested and observed token lengths, achieved cache fractions, sampled device memory and available active/queued observations.

Show all three repetition values and their spread. Use p50 and range for small samples. Do not claim p95 with fewer than 20 requests; a small set of correlated bursts does not establish robust tail latency. Token throughput is not correct-task goodput.

## Retain operational and evidence protections

Use the existing exclusive node owner, bounded watchdog, exact container/process identities, resource floors, request deadlines, cancellation and cleanup. Preserve failed attempts and their original retry ceilings. A changed schema or reporting scope never resets an attempt counter. Stop on memory/disk safety failures, worker/GPU faults, malformed transport or invalid encoding. Preserve evidence before any controlled recovery. This contract grants no VM or boot-disk lifecycle authority.

Every required cell needs three supported measured repetitions or an independently supported terminal boundary disposition. Missing work, budget omissions and unresolved invalid attempts remain incomplete. Preserve old pilots, tuning and other historical observations under their original scope; use fresh scored repetitions here. Only matched evidence supports comparisons. Independently verify raw records, coverage, failure denominators and scope-limited claims before preparing a sanitized report. Public publication requires separate approval.
