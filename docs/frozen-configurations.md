<!-- register: reproducibility guide | reader: inference benchmark reproducer | consumed: configuration verification -->
# Frozen configurations for the four-layout comparison

The [profile records](../configs/frozen-profiles/) preserve the exact four registered configurations and their qualification fingerprints. Their image, checkpoint and tokenizer pins apply to the current 336-cell comparison. Accepted scored coverage is reported separately from these qualified configuration records.

| Layout | Engines | Attention per engine | Expert parallelism per engine | Configured `max_num_seqs` | Attention backend | KV format |
| --- | --- | --- | --- | --- | --- | --- |
| `2xtp4` | 2 | TP4 | Disabled | 64 | `FLASHINFER_MLA_SPARSE_DSV41` | `fp8` |
| `4xdep2` | 4 | DP2/TP1 | EP2 | 32 | `FLASHINFER_MLA_SPARSE_DSV41` | `fp8` |
| `2xdep4` | 2 | DP4/TP1 | EP4 | 64 | `FLASHINFER_MLA_SPARSE_DSV41` | `fp8` |
| `dep8` | 1 | DP8/TP1 | EP8 | 128 | `FLASHMLA_MEGA_ATTN_DSV41` | `fp8_ds_mla` |

All configurations use the 1,048,576-token sequence limit, 8,192-token batched-prefill budget, chunked prefill, prefix caching, 0.90 GPU memory utilization, MXFP4 indexer storage and CPU Engram offload. Expert-parallel configurations enable embedding across data-parallel ranks and the FlashInfer autotuned DeepGEMM MegaMoE path. The frozen profiles disable speculation. These are whole-configuration comparisons: backend and KV-format differences prevent attributing a result to topology alone.

The serving image is `vllm/vllm-openai@sha256:aeb580d55ee3dcccd7b5c78d7871326f367bee5cfd8ca8e411c253d7983a0046`. The model and tokenizer use revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. The pinned router is version 0.1.14 with consistent-hash placement and inference retries disabled. Full dependency pins remain in the repository configuration files.

## Profile templates use identity placeholders

A profile hash covers a normalized template: its example image, model/cache mount sources, run identity and CPU list are placeholders. The record's top-level image and revision fields hold the measured pins. Preserve the template bytes when verifying its profile fingerprint. Bind actual local model/cache paths, a unique owned run identity, the pinned image and the discovered CPU partition through the supervised launcher.

The measured CPU policy reserves four whole physical cores on each of two NUMA nodes for clients, including both SMT threads, and assigns the remaining CPUs to the engines. The changed measured capture method uses one owned client subprocess inside that allocation. `cpu_partition.py` discovers the CPU identities; the template's `0,1` list does not describe the measured server allocation.

The following command renders the engine and router argument structure for one configuration. It starts no processes. Replace the model/cache paths with local paths and use the supervised campaign entry point for owned execution, readiness, monitoring, capture and cleanup.

```sh
python3 -B reproduce/launcher.py --layout 2xtp4 --image vllm/vllm-openai@sha256:aeb580d55ee3dcccd7b5c78d7871326f367bee5cfd8ca8e411c253d7983a0046 --run-id reproduction-preview --model-dir /models/DeepSeek-V4.1-Flash-1017 --cache-dir /cache/dsv41
```

The shared measured route is streamed Chat Completions, thinking disabled, temperature and top-p 1.0, exact fixed output length, final usage and no inference retries. The [comparison contract](comparison-contract.md) defines prompt matching, excluded warmup, cold namespaces, warm conditioning, finite-burst timing and evidence acceptance. Keep placement and achieved cache-hit observations beside results; a requested concurrency value includes queued work.

The configuration records retain qualification source seals and verifier hashes. They establish the selected profiles and their retained qualification references. Fresh scored repetitions, raw metric recomputation and complete coverage remain separately verifiable outputs.
