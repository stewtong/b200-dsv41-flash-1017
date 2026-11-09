<!-- register: technical reproduction guide | reader: benchmark reviewer | consumed: execute local CPU replay and inspect pinned recipe -->
# Reproduce the calculations and distinguish them from GPU replication

The candidate includes prompt-free request timings, token counts, retained evaluator outcomes and figure code. The arithmetic checks use only the standard library and passed on Python 3.9.6. The plotting environment used Python 3.13.13 with Matplotlib 3.10.7. Run the arithmetic check from the package root:

```sh
python3 reproduce/replay_tables.py
```

The check covers 2,290 load request records, 32 arm/window/profile cohorts, 20 natural block pairs and 880 task/seed/arm quality outcomes. It compares regenerated counts, rates, percentiles and conditional quality intervals with the exported summaries. A failed comparison exits unsuccessfully.

To regenerate the four figures, use Python 3.13 with the pinned plotting dependency:

```sh
dsv41_repro_dir="$(mktemp -d)"
python3.13 -m venv "$dsv41_repro_dir/env"
"$dsv41_repro_dir/env/bin/python" -m pip install -r reproduce/requirements.txt
"$dsv41_repro_dir/env/bin/python" reproduce/figures.py --output "$dsv41_repro_dir/figures"
```

The figure code reads CSV files and writes four PNG/SVG pairs in the scratch output directory. The environment and regenerated figures stay outside the sealed result tree. It makes no inference requests. Plotting was verified with Matplotlib 3.10.7; font and metadata differences can change image bytes across environments.

## The retained archive enables a deeper calculation audit

The separately retained v3 evidence archive is 18,982,251 bytes with SHA256:

```text
2fea0f8c0fba42bbfa4692c410d27b5b7d471fa47f74c3c4d60bbc4e7a5f23cc
```

That private archive retains frozen schedules, task selection and evaluation inputs, source, licenses, sanitized response samples and campaign accounting. Source and input hashes appear in `data/provenance.json`. Operational capture names are replaced by stable labels in the public timing table; measured values remain unchanged.

The public command above checks arithmetic on the included prompt-free records. It cannot independently score answers without the omitted prompts, model outputs and evaluator inputs. The retained-source analysis and representative response checks were performed separately on the exact archive; they are not an additional self-contained command in this public export. Original coding outcomes remain bound to the isolated scorer, without executing generated code during CPU review.

Fresh GPU replication requires installing the pinned runtime, preparing checkpoint/cache/router inputs and repeating capture on hardware. The checked-in harness has its own operator prerequisites, described in [the repository reproduction notes](../../docs/reproduction-notes.md). This report demonstrates arithmetic replay and reports the retained-evidence checks; it does not demonstrate fresh-node replication. Archive access or redistribution requires a separate owner decision.

## The recorded serving recipe

The report supplies the hardware, checkpoint revision, image digest, vLLM source revision and compute-stack versions. The following settings supplement that inventory. They describe the measured recipe, rather than a production deployment command.

| Setting | Recorded value |
| --- | --- |
| GPU placement | Two TP4 engines on GPU IDs 0 to 3 and 4 to 7, on one eight-B200 node |
| CPU placement evidence | Representative natural and load NoSpec launch records reserve logical CPUs 0 to 7 and 80 to 87 for the client; both engine containers share CPUs 8 to 79 and 88 to 159. This is a two-NUMA client/server partition, not per-replica NUMA pinning. It is not an independent audit of every K5 launch. |
| Frontend and runner | `VLLM_USE_RUST_FRONTEND=1`, `VLLM_USE_V2_MODEL_RUNNER=1` |
| KV cache | `--kv-cache-dtype fp8` |
| Configured limits | Model length 1,048,576; 64 sequences per engine; 8,192 batched tokens; GPU memory utilization 0.90 |
| Prefill/cache | Chunked prefill and prefix caching enabled; cache reuse/history not causally isolated |
| Parsing | `deepseek_v41` tokenizer mode, reasoning parser and tool-call parser; auto tool choice enabled |
| Compilation | `VLLM_COMPILE`, `FULL_AND_PIECEWISE` CUDA graphs |
| Engram | CPU offload enabled |
| Attention | `FLASHINFER_MLA_SPARSE_DSV41`, indexer KV dtype `mxfp4`, sparse indexer logits enabled |
| K5 change | Five speculative tokens, probabilistic draft sampling, block rejection enabled, adaptive verification disabled |
| NoSpec change | Speculation disabled; common topology/runtime recipe retained |
| Router | vLLM router 0.1.14, `consistent_hash`, intra-node data-parallel size 1, request timeout 14,400 seconds, retries disabled |
| Routing surface | Node-local loopback engines on ports 31000/31001 and router on 31100; stable per-conversation session identities supplied by capture |

Configured model length and sequence limits are settings, not demonstrated capacity at those joint limits. The benchmark did not qualify 64 simultaneous million-token contexts. Existing compatible caches were reused; no clean cold-cache or isolated-kernel causal claim follows.

## Request settings differ by workload

All listed requests use streaming with usage included, temperature 1.0 and top-p 1.0. Fast uses `chat_template_kwargs.thinking=false`; Think uses `true`. Request bodies, seeds and selection rules are frozen in the retained schedules. Quality seeds are 19 and 43, transformed deterministically with task identity and conversation turn by the retained encoder. Other workloads have their own frozen seeds.

| Workload/profile | Numeric reasoning effort | Maximum output tokens | Natural stop |
| --- | --- | --- | --- |
| Natural performance, Fast | 25 | 256 | Yes |
| Quality Fast coding | 25 | 1,024 | Yes |
| Quality Fast grounded | 25 | 128 | Yes |
| Quality Fast reasoning | 25 | 512 | Yes |
| Quality Fast tools | 25 | 256 per turn | Yes; first turn requires one inert tool call |
| Quality Think, all families | 25 | 1,024 per turn | Yes |
| Mixed load Fast grounded/reasoning | 25 | 128 / 512 | Yes |
| Mixed load Think grounded/reasoning | 25 | 1,024 | Yes |
| Fixed-output anchor, Fast | 50 | 2,048 | No: `min_tokens=2048`, `ignore_eos=true` |

The small natural outputs in the report are actual returned counts, not these caps. The load panel repeatedly cycles five reasoning and five answerable grounded questions, each in Fast and Think form. The fixed-output anchor is a separate synthetic task. Quality task selection was fixed before model scoring. Load selection used the recorded schedule order without filtering on model outcome; the preparation read existing NoSpec quality evidence.

## Definitions needed to interpret replay

Load request clocks are integer nanoseconds relative to each window's start. The window is `[0, 60 seconds)`. Completion inside the window and correctness are separate tests. Deadline goodput additionally requires dispatch-to-completion latency at most 1, 5, 15 or 60 seconds. All four predeclared sensitivity limits are exported; none is a production SLA.

First-visible latency runs from actual dispatch until the first nonempty visible-content event. It includes elapsed time spent on preceding hidden reasoning, although reasoning deltas do not terminate the interval. It is not independently timestamped first-token arrival. Reported p95 uses the sorted value at zero-based index `ceil((n-1)*0.95)`, without interpolation. `load-windows.csv` supplies the all-visible and correctness-successful/in-window cohort sizes beside both tails. The report plots the all-visible Fast cohort.

Natural token throughput divides all transport/content-valid output tokens by recorded burst duration, including incorrect answers. Each of four regimes has five prompt-block ratios. The nominal block-bootstrap intervals are descriptive and conditional on those blocks; their endpoints equal the observed extremes. There are no independent deployment replicates.

Quality differences are K5 minus NoSpec, paired by task ID, profile and seed. The task-ID bootstrap uses 10,000 draws with random seed 20261108 and carries both seed outcomes together. It does not stratify families, account for shared tool-template clusters or resample deployments. The small Think subset and public-task contamination limit the inference. CSV replay accepts retained coding scorer outcomes; it does not execute generated code or establish general quality equivalence.

## Dataset attribution and release scope

The study uses selected tasks from [HumanEval](https://github.com/openai/human-eval), [GSM8K](https://github.com/openai/grade-school-math) and [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/), plus synthetic inert-tool tasks. These are custom selections and evaluators. HumanEval and GSM8K license texts are retained in the full archive; SQuAD's source site identifies its CC BY-SA 4.0 dataset terms. Source texts, prompts, generated programs and full raw captures are omitted from this candidate. Dataset licenses do not grant rights to the model or serving software; their separate terms apply to a hardware reproduction.

The selected-panel, prompt-free release permits arithmetic scrutiny while preserving the distinction from raw-oracle and hardware replication. Full sustained-service claims remain unsupported because drain queue sampling is incomplete. That limitation cannot be repaired by replaying these tables.
