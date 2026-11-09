<!-- register: data dictionary | reader: benchmark reviewer | consumed: inspect exported measurements -->
# Tables distinguish timing, correctness and repeated task identity

All files are UTF-8 CSV. Empty timing values mean no recorded clock. Task IDs identify repeated bodies, not independent arrivals or unique questions. Ratios are descriptive comparisons of the recorded recipes.

| File | Row unit and key fields |
| --- | --- |
| `natural-requests.csv` | One observed natural request per arm/block/input target/concurrency/index. `output_tokens` and request/first-visible seconds are recorded; `group_seconds` is the shared burst duration. |
| `natural-blocks.csv` | One paired block/regime. `token_throughput_ratio=(K5 tokens/K5 group seconds)/(NoSpec tokens/NoSpec group seconds)`. `tokens_ratio` and `inverse_duration_ratio` decompose that quantity. Task-success counts are separate. |
| `natural-regimes.csv` | Median and nominal conditional block-bootstrap endpoints over five pairs per regime. |
| `load-requests.csv` | One scheduled load request, with arm/window/profile/task ID/index; clocks relative to window start in integer nanoseconds; retained success/category and output tokens. No prompt or response text. |
| `load-windows.csv` | One arm/window/profile cohort; `all` combines Fast and Think. `successful_*` means task-correct, not merely HTTP-successful. In-window counts use the half-open 60-second window. Both first-visible populations and sample counts are explicit; successful end-to-end p95 uses correct in-window completions only. `backlog_at_end` is client completion accounting, not a sampled server-queue length. |
| `load-pairs.csv` | One matched window/profile pair. `successful_rate_ratio` and `e2e_1s_goodput_ratio` are K5/NoSpec. `first_visible_p95_nospec_over_k5` is explicitly NoSpec/K5, so values above one indicate a lower K5 tail. |
| `load-deadlines.csv` | Arm/window/profile/limit; correct requests completed inside the window and within the dispatch-to-completion limit. All four predeclared limits are included. |
| `load-tasks.csv` | Arm/window/task ID counts; distinguishes in-window correctness from correctness including the drain period. |
| `quality-outcomes.csv` | One arm/profile/task ID/seed retained evaluator outcome. Coding success is joined to the isolated scorer by task/profile/seed identity. Repeated seed outcomes stay together in the bootstrap. |
| `quality.csv` | Selected-panel success totals and K5-minus-NoSpec percentage-point difference with nominal conditional task-ID bootstrap endpoints. |
| `provenance.json` | SHA256 binding to the separate v3 archive and accepted natural/quality/scorer analysis inputs; documents capture-label replacement. |

The [reproduction guide](REPRODUCE.md) defines p95, window membership, bootstrap dependence and the limits of CPU replay. Published aggregate totals can be replayed without inference; task correctness cannot be independently re-scored from prompt-free outcome CSVs alone.
