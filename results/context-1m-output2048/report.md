<!-- register: benchmark supplement | reader: serving engineer | consumed: output-length comparison -->
# 1M-input output-length supplement

The original 512-output campaign and the 2,048-output supplement are shown separately. The original panel used a nominal 1M input target with its recorded calibration tolerance.
The supplement accepted 44 of 44 cells, with 132 verified repetitions and 3552 verified requests.
Rates are median output tokens per second across three finite-burst repetitions; minimum and maximum show repetition spread.

## 512 requested output tokens

| Layout | Cache | Concurrency | Verdict | Repetitions | Requests | Median | Min | Max |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 2xtp4 | cold | 1 | accepted | 3 | 3 | 29.372 | 29.343 | 30.997 |
| 2xtp4 | cold | 2 | accepted | 3 | 6 | 61.794 | 61.475 | 61.895 |
| 2xtp4 | cold | 4 | accepted | 3 | 12 | 56.230 | 55.424 | 56.239 |
| 2xtp4 | cold | 8 | accepted | 3 | 24 | 53.828 | 53.797 | 53.867 |
| 2xtp4 | cold | 16 | accepted | 3 | 48 | 52.834 | 52.467 | 53.028 |
| 2xtp4 | cold | 32 | accepted | 3 | 96 | 52.015 | 51.819 | 52.145 |
| 2xtp4 | cold | 64 | accepted | 3 | 192 | 51.179 | 51.165 | 51.495 |
| 2xtp4 | cold | 128 | accepted | 3 | 384 | 51.921 | 51.070 | 52.151 |
| 2xtp4 | warm_prefix | 1 | accepted | 3 | 3 | 216.134 | 215.528 | 216.932 |
| 2xtp4 | warm_prefix | 8 | accepted | 3 | 24 | 1108.918 | 1073.601 | 1122.371 |
| 2xtp4 | warm_prefix | 32 | accepted | 3 | 96 | 1990.995 | 1900.594 | 2075.584 |
| 4xdep2 | cold | 1 | accepted | 3 | 3 | 19.637 | 19.603 | 20.274 |
| 4xdep2 | cold | 2 | accepted | 3 | 6 | 42.815 | 42.725 | 44.643 |
| 4xdep2 | cold | 4 | accepted | 3 | 12 | 88.580 | 85.605 | 88.817 |
| 4xdep2 | cold | 8 | accepted | 3 | 24 | 176.400 | 176.280 | 176.635 |
| 4xdep2 | cold | 16 | accepted | 3 | 48 | 189.972 | 189.190 | 190.301 |
| 4xdep2 | cold | 32 | accepted | 3 | 96 | 199.111 | 198.805 | 199.288 |
| 4xdep2 | cold | 64 | accepted | 3 | 192 | 203.675 | 202.723 | 203.781 |
| 4xdep2 | cold | 128 | accepted | 3 | 384 | 205.899 | 205.769 | 206.065 |
| 4xdep2 | warm_prefix | 1 | accepted | 3 | 3 | 139.936 | 139.478 | 140.046 |
| 4xdep2 | warm_prefix | 8 | accepted | 3 | 24 | 975.587 | 961.165 | 975.899 |
| 4xdep2 | warm_prefix | 32 | accepted | 3 | 96 | 2358.474 | 2321.997 | 2406.507 |
| 2xdep4 | cold | 1 | accepted | 3 | 3 | 22.611 | 22.451 | 22.753 |
| 2xdep4 | cold | 2 | accepted | 3 | 6 | 38.133 | 37.300 | 38.716 |
| 2xdep4 | cold | 4 | accepted | 3 | 12 | 86.496 | 84.991 | 86.601 |
| 2xdep4 | cold | 8 | accepted | 3 | 24 | 171.300 | 170.392 | 171.373 |
| 2xdep4 | cold | 16 | accepted | 3 | 48 | 184.917 | 184.096 | 185.015 |
| 2xdep4 | cold | 32 | accepted | 3 | 96 | 193.396 | 192.839 | 194.006 |
| 2xdep4 | cold | 64 | accepted | 3 | 192 | 196.981 | 196.800 | 197.199 |
| 2xdep4 | cold | 128 | accepted | 3 | 384 | 199.633 | 199.411 | 199.928 |
| 2xdep4 | warm_prefix | 1 | accepted | 3 | 3 | 146.809 | 145.704 | 147.295 |
| 2xdep4 | warm_prefix | 8 | accepted | 3 | 24 | 974.495 | 946.482 | 994.470 |
| 2xdep4 | warm_prefix | 32 | accepted | 3 | 96 | 2303.281 | 2264.119 | 2346.525 |
| dep8 | cold | 1 | accepted | 3 | 3 | 25.104 | 24.959 | 25.181 |
| dep8 | cold | 2 | accepted | 3 | 6 | 44.122 | 39.697 | 44.973 |
| dep8 | cold | 4 | accepted | 3 | 12 | 71.726 | 71.614 | 72.803 |
| dep8 | cold | 8 | accepted | 3 | 24 | 173.036 | 172.946 | 173.643 |
| dep8 | cold | 16 | accepted | 3 | 48 | 189.119 | 185.585 | 189.758 |
| dep8 | cold | 32 | accepted | 3 | 96 | 199.231 | 198.861 | 199.666 |
| dep8 | cold | 64 | accepted | 3 | 192 | 204.661 | 204.639 | 204.709 |
| dep8 | cold | 128 | accepted | 3 | 384 | 207.398 | 206.802 | 207.556 |
| dep8 | warm_prefix | 1 | accepted | 3 | 3 | 137.511 | 137.299 | 140.030 |
| dep8 | warm_prefix | 8 | accepted | 3 | 24 | 883.822 | 879.370 | 888.243 |
| dep8 | warm_prefix | 32 | accepted | 3 | 96 | 2060.975 | 2007.970 | 2172.794 |

## 2048 requested output tokens

| Layout | Cache | Concurrency | Verdict | Repetitions | Requests | Median | Min | Max |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 2xtp4 | cold | 1 | accepted | 3 | 3 | 82.983 | 82.942 | 89.483 |
| 2xtp4 | cold | 2 | accepted | 3 | 6 | 179.286 | 179.180 | 179.341 |
| 2xtp4 | cold | 4 | accepted | 3 | 12 | 187.491 | 182.087 | 187.790 |
| 2xtp4 | cold | 8 | accepted | 3 | 24 | 193.597 | 190.976 | 195.129 |
| 2xtp4 | cold | 16 | accepted | 3 | 48 | 195.769 | 195.517 | 198.695 |
| 2xtp4 | cold | 32 | accepted | 3 | 96 | 201.494 | 201.075 | 201.783 |
| 2xtp4 | cold | 64 | accepted | 3 | 192 | 203.610 | 203.283 | 204.320 |
| 2xtp4 | cold | 128 | accepted | 3 | 384 | 204.597 | 204.232 | 204.605 |
| 2xtp4 | warm_prefix | 1 | accepted | 3 | 3 | 238.437 | 238.131 | 238.946 |
| 2xtp4 | warm_prefix | 8 | accepted | 3 | 24 | 1298.908 | 1253.061 | 1422.292 |
| 2xtp4 | warm_prefix | 32 | accepted | 3 | 96 | 3283.910 | 3098.394 | 3429.781 |
| 4xdep2 | cold | 1 | accepted | 3 | 3 | 55.282 | 49.801 | 55.602 |
| 4xdep2 | cold | 2 | accepted | 3 | 6 | 117.406 | 117.173 | 122.284 |
| 4xdep2 | cold | 4 | accepted | 3 | 12 | 243.592 | 233.209 | 244.026 |
| 4xdep2 | cold | 8 | accepted | 3 | 24 | 485.037 | 484.555 | 485.575 |
| 4xdep2 | cold | 16 | accepted | 3 | 48 | 604.198 | 561.268 | 604.669 |
| 4xdep2 | cold | 32 | accepted | 3 | 96 | 690.001 | 674.499 | 690.123 |
| 4xdep2 | cold | 64 | accepted | 3 | 192 | 706.403 | 704.235 | 706.403 |
| 4xdep2 | cold | 128 | accepted | 3 | 384 | 733.379 | 730.543 | 734.005 |
| 4xdep2 | warm_prefix | 1 | accepted | 3 | 3 | 149.495 | 149.238 | 149.629 |
| 4xdep2 | warm_prefix | 8 | accepted | 3 | 24 | 1100.052 | 1092.931 | 1122.314 |
| 4xdep2 | warm_prefix | 32 | accepted | 3 | 96 | 3125.048 | 2958.637 | 3243.008 |
| 2xdep4 | cold | 1 | accepted | 3 | 3 | 61.923 | 61.884 | 62.574 |
| 2xdep4 | cold | 2 | accepted | 3 | 6 | 110.611 | 106.994 | 110.731 |
| 2xdep4 | cold | 4 | accepted | 3 | 12 | 239.001 | 231.448 | 239.543 |
| 2xdep4 | cold | 8 | accepted | 3 | 24 | 473.076 | 471.746 | 474.974 |
| 2xdep4 | cold | 16 | accepted | 3 | 48 | 585.621 | 554.364 | 586.064 |
| 2xdep4 | cold | 32 | accepted | 3 | 96 | 670.920 | 635.703 | 672.659 |
| 2xdep4 | cold | 64 | accepted | 3 | 192 | 725.081 | 724.730 | 726.529 |
| 2xdep4 | cold | 128 | accepted | 3 | 384 | 759.331 | 756.344 | 760.048 |
| 2xdep4 | warm_prefix | 1 | accepted | 3 | 3 | 155.847 | 155.350 | 157.074 |
| 2xdep4 | warm_prefix | 8 | accepted | 3 | 24 | 1099.210 | 1094.938 | 1111.765 |
| 2xdep4 | warm_prefix | 32 | accepted | 3 | 96 | 3133.559 | 3108.453 | 3155.158 |
| dep8 | cold | 1 | accepted | 3 | 3 | 66.087 | 66.022 | 67.241 |
| dep8 | cold | 2 | accepted | 3 | 6 | 121.080 | 118.584 | 123.478 |
| dep8 | cold | 4 | accepted | 3 | 12 | 207.137 | 206.657 | 208.778 |
| dep8 | cold | 8 | accepted | 3 | 24 | 465.984 | 465.912 | 466.012 |
| dep8 | cold | 16 | accepted | 3 | 48 | 589.707 | 576.150 | 589.752 |
| dep8 | cold | 32 | accepted | 3 | 96 | 684.706 | 656.659 | 687.372 |
| dep8 | cold | 64 | accepted | 3 | 192 | 749.537 | 747.910 | 749.579 |
| dep8 | cold | 128 | accepted | 3 | 384 | 785.957 | 785.144 | 788.156 |
| dep8 | warm_prefix | 1 | accepted | 3 | 3 | 145.471 | 145.046 | 146.057 |
| dep8 | warm_prefix | 8 | accepted | 3 | 24 | 1005.173 | 997.335 | 1016.822 |
| dep8 | warm_prefix | 32 | accepted | 3 | 96 | 3127.415 | 3075.658 | 3177.542 |

## Interpretation

Each accepted supplement request has exactly 1,000,000 rendered input tokens and 2,048 output tokens. The two output-length panels use different requested output lengths and are not interchangeable.
Finite bursts do not establish sustained capacity or task-quality equivalence. Cold-cache reset, configuration-major temporal drift, sampling, repetition spread, stream chunk timing, and achieved warm-prefix hit rates limit comparisons. Equal requested output within the supplement removes one confound, not all confounds.
