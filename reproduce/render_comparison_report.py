"""Render descriptive charts from independently recomputable sanitized tables."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

import comparison_scope
import runtime
import verify_comparison_tables


COLORS = dict(zip(comparison_scope.LAYOUTS, ('#1768ac', '#d46a16', '#20836b', '#8459a5')))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def populations(tables):
    """Return complete cells only; missing measurements never become zeros."""
    cells = rows(tables / 'cells.csv')
    observed = {}
    for row in rows(tables / 'repetitions.csv'):
        if row['observed'] == 'True':
            observed.setdefault(row['id'], []).append(row)
    result = {}
    for cell in cells:
        if cell['accepted'] != 'True':
            continue
        reps = observed[cell['id']]
        if len(reps) != 3:
            raise ValueError('complete chart cell requires three repetitions')
        key = (cell['cache'], int(cell['input_tokens']), cell['layout'], int(cell['concurrency']))
        result[key] = reps
    return result


def interval(repetitions, metric):
    if repetitions is None:
        return (math.nan,) * 3
    values = [float(r[metric]) for r in repetitions if r[metric] != '']
    if len(values) != 3:
        return (math.nan,) * 3
    return statistics.median(values), min(values), max(values)


def render(tables, output):
    tables = Path(tables).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('select a fresh report directory')
    checked = verify_comparison_tables.verify(tables)
    source_hashes = {p.name: sha(p) for p in tables.iterdir()}
    summary = json.loads((tables / 'summary.json').read_text())
    measured = populations(tables)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.comparison-report-', dir=output.parent) as directory:
        stage = Path(directory) / 'report'
        stage.mkdir()
        os.environ['MPLCONFIGDIR'] = str(Path(directory) / 'matplotlib-config')
        os.environ['XDG_CACHE_HOME'] = str(Path(directory) / 'cache')
        Path(os.environ['XDG_CACHE_HOME']).mkdir()
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D
        matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
            'axes.spines.top': False, 'axes.spines.right': False,
            'svg.fonttype': 'none', 'svg.hashsalt': 'dsv41-fixed-comparison-v1'})
        specifications = [
            ('cold-burst-rate', 'cold', 'burst_output_tokens_per_second',
             'Burst output rate (tokens/s)', 'Median and range of three repetition burst rates.', False),
            ('cold-first-output', 'cold', 'first_output_seconds_p50',
             'First output latency (seconds)', 'Median and range of three per-repetition request medians.', True),
            ('cold-completion-latency', 'cold', 'completion_seconds_p50',
             'Completion latency (seconds)', 'Median and range of three per-repetition request medians.', True),
            ('warm-burst-rate', 'warm_prefix', 'burst_output_tokens_per_second',
             'Warm-prefix burst output rate (tokens/s)', 'Median and range of three repetition burst rates.', False),
            ('warm-achieved-cache-hit', 'warm_prefix', 'achieved_cache_hit_fraction',
             'Achieved cache-hit fraction', 'Median and range of three repetition native counter fractions.', False),
        ]
        figures = []
        for name, cache, metric, ylabel, note, logarithmic in specifications:
            inputs = comparison_scope.INPUTS if cache == 'cold' else comparison_scope.WARM_INPUTS
            cs = comparison_scope.CONCURRENCIES if cache == 'cold' else comparison_scope.WARM_CONCURRENCIES
            columns = 3 if cache == 'cold' else 2
            fig, axes = plt.subplots(math.ceil(len(inputs) / columns), columns,
                figsize=(12, 10 if cache == 'cold' else 7), squeeze=False)
            for ax, n in zip(axes.flat, inputs):
                observed_cells = 0
                accepted_cells = 0
                for layout in comparison_scope.LAYOUTS:
                    points = [interval(measured.get((cache, n, layout, c)), metric) for c in cs]
                    accepted_cells += sum((cache, n, layout, c) in measured for c in cs)
                    mid, low, high = zip(*points)
                    observed_cells += sum(math.isfinite(value) for value in mid)
                    ax.plot(cs, mid, marker='o', markersize=3.5, linewidth=1.5,
                        color=COLORS[layout], label=layout)
                    ax.fill_between(cs, low, high, color=COLORS[layout], alpha=.13, linewidth=0)
                ax.set_xscale('log', base=2)
                ax.set_xticks(cs, [str(c) for c in cs])
                ax.set_xlim(cs[0] / 1.15, cs[-1] * 1.15)
                ax.set_title(f'{n:,} input / {512 if n == 1000000 else 2048:,} output tokens', loc='left', fontsize=10)
                ax.set_xlabel('Node-wide outstanding requests')
                ax.set_ylabel(ylabel)
                ax.grid(axis='y', color='#dddddd', linewidth=.5)
                if observed_cells:
                    if logarithmic:
                        ax.set_yscale('log')
                    elif metric == 'achieved_cache_hit_fraction':
                        ax.set_ylim(0, 1)
                    else:
                        ax.set_ylim(bottom=0)
                else:
                    ax.set_ylim(0, 1)
                    ax.set_yticks([])
                    label = 'No complete cells' if not accepted_cells else 'No defined values'
                    ax.text(.5, .5, label, transform=ax.transAxes,
                        ha='center', va='center', color='#737373')
                ax.text(.02, .97, f'{accepted_cells}/{4 * len(cs)} accepted; {observed_cells} shown',
                    transform=ax.transAxes, va='top', fontsize=8, color='#656565')
            for unused in list(axes.flat)[len(inputs):]:
                unused.set_visible(False)
            fig.suptitle(ylabel + f' | accepted table coverage {summary["accepted_cells"]}/336',
                x=.06, ha='left', fontsize=15)
            handles = [Line2D([], [], color=COLORS[layout], marker='o', markersize=4,
                label=layout) for layout in comparison_scope.LAYOUTS]
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .955), ncol=4, frameon=False)
            fig.text(.06, .035, note + '\nOnly complete cells are shown. Curves compare fixed configurations; no winner is inferred.', fontsize=9)
            fig.tight_layout(rect=(.015, .085, .995, .9), h_pad=2, w_pad=2)
            for extension in ('svg', 'png'):
                fig.savefig(stage / f'{name}.{extension}', dpi=150,
                    metadata={'Date': None} if extension == 'svg' else {'Software': 'Matplotlib'})
            plt.close(fig)
            figures.append(dict(name=name, cache=cache, metric=metric, note=note,
                complete_cells_in_cache=sum(key[0] == cache for key in measured)))
        matching_limit = ('All required configurations are matched; repetition variation and configuration-major time windows still limit rankings.'
            if summary['complete'] and summary['all_four_matched_cells'] == 336 else
            'Incomplete matching and repetition variation limit rankings; configuration-major time windows add temporal uncertainty.')
        text = f'''<!-- register: benchmark report | reader: serving engineer | consumed: result and chart inspection -->
# Descriptive fixed-configuration comparison

The tables contain {summary['accepted_cells']} accepted cells of 336: {summary['accepted_cold_cells']}/288 cold and {summary['accepted_warm_cells']}/48 warm-prefix cells. They retain {summary['measured_repetitions']} measured repetitions and {summary['measured_requests']:,} requests. Four-configuration matched coverage is {summary['all_four_matched_cells']} cells. Table coverage complete: {str(summary['complete']).lower()}.

The charts use complete three-repetition cells only. Missing measurements stay blank and break curves. Shading shows the range across repetitions. Latency charts show the median and range of three per-repetition request medians, without pooling different repetitions. They do not show p95 or robust tail latency. Node-wide outstanding demand includes queued requests.

Cold measurements use cache-busted namespaces; cache flush or reset was not verified. Warm-designated misses stay in warm results with achieved native-counter hit fractions. A missing fraction is left blank. The one-million-input row requests 512 output tokens; other rows request 2,048.

This report presents finite bursts for the four frozen configurations. Necessary backend and KV-format differences are part of those configurations. It does not establish topology causation, optimal tuning, sustained capacity, task-quality equivalence or production reliability. {matching_limit}

The portable table verifier recomputes timing and usage arithmetic before rendering. This rendering step grants no new raw-stream, profile, placement or telemetry acceptance, and does not establish final campaign completion. Original source-bound capture audits and the final campaign audit remain separate evidence.

## Cold burst rate

![Cold burst rate](cold-burst-rate.svg)

## First output latency

![Cold first output latency](cold-first-output.svg)

## Completion latency

![Cold completion latency](cold-completion-latency.svg)

## Warm-prefix burst rate

![Warm-prefix burst rate](warm-burst-rate.svg)

## Achieved warm-prefix cache reuse

![Achieved cache-hit fraction](warm-achieved-cache-hit.svg)

## Reproduce the figures

Install the pinned report requirements in an isolated Python environment, then run `python3 -B reproduce/render_comparison_report.py TABLE_DIRECTORY FRESH_OUTPUT_DIRECTORY`. The input is a sealed sanitized table directory, not an unverified result summary. `provenance.json` records source fingerprints and the generator and plotting versions. Existing output directories are rejected. A failed input check leaves prior reports unchanged.
'''
        (stage / 'README.md').write_text(text)
        runtime.atomic_json(stage / 'provenance.json', dict(status='descriptive_report_rendered',
            table_verification=checked, source_files_sha256=source_hashes,
            generator_sha256=sha(Path(__file__)), matplotlib_version=matplotlib.__version__,
            figures=figures, new_measurement_acceptance=False))
        runtime.seal_evidence(stage)
        runtime.verify_evidence_manifest(stage)
        if {p.name: sha(p) for p in tables.iterdir()} != source_hashes:
            raise ValueError('source tables changed during rendering')
        if output.exists() or output.is_symlink():
            raise FileExistsError('report output appeared during rendering')
        stage.rename(output)
    print(json.dumps(dict(status='descriptive_report_rendered', output=str(output),
        accepted_cells=checked['accepted_cells'], complete=checked['complete'], figures=len(figures))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tables', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.tables, args.output)
