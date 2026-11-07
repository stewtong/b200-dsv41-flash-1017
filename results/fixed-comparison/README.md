<!-- register: benchmark evidence | reader: serving engineer | consumed: coverage and reproduction -->
# Complete fixed-configuration comparison

All 336 required cells were accepted: 288 cold and 48 warm-prefix cells, with 1,008 repetitions and 29,508 measured requests. Each cell has three repetitions. The [report](report/README.md) plots the median and range across repetitions; the [tables](tables/) support timing and usage recomputation.

The four configurations use eight B200 GPUs in total: two TP4 replicas, four DEP2 replicas, two DEP4 replicas or one DEP8 replica. Context targets span 8,000 to 1,000,000 input tokens. Cold bursts contain 1 to 128 outstanding requests; the warm-prefix panel uses 1, 8 and 32. Original 1M-input cells request 512 output tokens; other contexts request 2,048. The separate [1M-input supplement](../context-1m-output2048/report.md) uses 2,048 output tokens.

The [scientific configuration](reproduce/scientific-configuration.json) records the complete matrix, frozen profile templates and image, model, tokenizer and router pins. [Original source provenance](reproduce/original-source-provenance.json) identifies historical producer versions by content hashes. Those historical archives and original raw evidence are retained privately. The portable harness in this repository has its own Git identity and does not replace the historical producer versions.

```sh
python3 -B reproduce/verify_result_package.py results/fixed-comparison
```

This verifies the complete public file set and hashes, then recomputes coverage and timing/usage arithmetic from the public request tables. It does not replay the original raw-stream, placement, telemetry, input-parent or lifecycle audits. Those separate audits established campaign acceptance before export.

The [publication contract](../../docs/comparison-contract.md) defines first-output timing, finite-burst output rate, cache conditions and repetition summaries. Cold namespaces were cache-busted; cache flush was not verified. Warm-designated misses remain included with achieved reuse. Configuration-major execution leaves temporal drift unresolved. Necessary backend and KV-format differences are part of the configuration comparison. Finite bursts establish descriptive observations; sustained capacity, general task-quality equivalence and production reliability require separate testing.

## Reproduce the figures

Install `reproduce/report-requirements.txt` into an isolated Python environment, then use a fresh output directory:

```sh
python3 -B reproduce/render_comparison_report_presentation.py results/fixed-comparison/tables /tmp/dsv41-report
```

The retained presentation renderer shortens the warm-chart axis label to prevent overlap. Source table bytes and plotted values are unchanged from the verified export. `report/provenance.json` records the exact generator and plotting versions used for the included figures.
