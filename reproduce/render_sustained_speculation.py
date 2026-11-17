"""Render deterministic SVGs from the published sustained-load summary."""
import argparse
from html import escape
import json
import math
from pathlib import Path


ARMS = {
    'A': ('2xTP4, DSpark K5', '#1765a8', -33),
    'D': ('2xTP4, NoSpec', '#20845a', -11),
    'B': ('DEP8, NoSpec', '#9a6400', 11),
    'C': ('DEP8, DSpark K5', '#d15e26', 33),
}
RATES = {'W1': (0.05, 0.15, 0.45, 1.35, 4.05), 'W2': (0.02, 0.08, 0.32)}
METRICS = {
    'acceptance.svg': ('acceptance_length', 'K5 acceptance by offered load',
                       'Acceptance length (tokens per draft step)'),
    'completion-p95.svg': ('request_completion_p95_seconds',
                           'Successful-turn completion p95 by offered load',
                           'Successful-turn completion p95 (seconds)'),
}
WIDTH, HEIGHT = 1500, 830
TOP, BOTTOM = 224, 552
PANEL_WIDTH = 602
PANEL_LEFT = {'W1': 105, 'W2': 850}
PURPLE, RED, GREY = '#7c2d8b', '#b62424', '#718096'


def text(x, y, value, size=14, **attributes):
    options = {'x': x, 'y': y, 'font-size': size, **attributes}
    attrs = ' '.join(f'{key}="{escape(str(value), quote=True)}"' for key, value in options.items())
    return f'<text {attrs}>{escape(str(value))}</text>'


def line(x1, y1, x2, y2, color='#d9e1e8', width=1):
    return (f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
            f'stroke="{color}" stroke-width="{width}"/>')


def marker(x, y, color, repeat=0, failed=False, diagnostic=False, missing=False):
    if diagnostic:
        return (f'<path d="M{x:.3f},{y - 7:.3f} L{x + 7:.3f},{y + 6:.3f} '
                f'L{x - 7:.3f},{y + 6:.3f} Z" fill="{PURPLE}" stroke="white" stroke-width="1"/>')
    if failed:
        return (f'<path d="M{x - 6:.3f},{y - 6:.3f} L{x + 6:.3f},{y + 6:.3f} '
                f'M{x + 6:.3f},{y - 6:.3f} L{x - 6:.3f},{y + 6:.3f}" '
                f'stroke="{RED}" stroke-width="2.5"/>')
    fill = 'white' if repeat else color
    if missing:
        return (f'<rect x="{x - 4:.3f}" y="{y - 4:.3f}" width="8" height="8" '
                f'fill="{fill}" stroke="{color}" stroke-width="2"/>')
    return (f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="{fill}" '
            f'stroke="{color}" stroke-width="2"/>')


def tick_scale(rows, key):
    values = [float(row[key]) for row in rows if row[key] is not None]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('chart values must be finite, nonnegative and nonempty')
    if key == 'acceptance_length':
        step = 0.1
        lower = math.floor(min(values) / step - 0.25) * step
        upper = math.ceil(max(values) / step + 0.25) * step
    else:
        lower = 0
        rough = max(values) / 6
        magnitude = 10 ** math.floor(math.log10(rough)) if rough else 1
        step = next(factor * magnitude for factor in (1, 2, 5, 10) if factor * magnitude >= rough)
        upper = max(step, math.ceil(max(values) / step) * step)
    ticks = [round(lower + index * step, 10) for index in range(round((upper - lower) / step) + 1)]
    return lower, upper, ticks


def all_rows(summary):
    rows = summary['cells'] + [summary['diagnostic']]
    if len(summary['cells']) != 39 or len({row['capture_id'] for row in rows}) != 40:
        raise ValueError('expected 39 distinct core captures plus one split diagnostic')
    for row in rows:
        if (row['arm'] not in ARMS or row['workload'] not in RATES or
                row['rate'] not in RATES[row['workload']] or row['repeat'] not in (0, 1)):
            raise ValueError('unknown chart category')
    if summary['diagnostic']['disposition'] != 'cache_confounded_diagnostic':
        raise ValueError('split diagnostic disposition changed')
    return rows


def render(summary, filename):
    rows = all_rows(summary)
    key, heading, ylabel = METRICS[filename]
    lower, upper, ticks = tick_scale(rows, key)

    def y_position(value):
        return BOTTOM - (value - lower) / (upper - lower) * (BOTTOM - TOP)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
             f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chart-title chart-description" '
             'font-family="Arial, Helvetica, sans-serif" fill="#172b3a">',
             '<!-- register: scientific result chart | reader: serving engineers | consumed: report figure -->',
             f'<title id="chart-title">{escape(heading)}</title>',
             '<desc id="chart-description">39 core windows plus split diagnostic. '
             'Offered rates are categorical, not a fitted trend. Numeric scales are shared across workloads. '
             'Unavailable values appear in a separate strip. Each capture has exact source values in data attributes.</desc>',
             '<rect width="100%" height="100%" fill="white"/>',
             text(50, 39, heading, 26, **{'font-weight': 'bold'}),
             text(50, 68, '39 core windows plus split diagnostic; 400 forced output tokens per turn.', 15)]
    for index, (arm, (label, color, _)) in enumerate(ARMS.items()):
        x = 62 + index * 365
        parts.extend([marker(x, 106, color), text(x + 15, 111, f'{arm}: {label}', 16)])
    parts.extend([
        marker(62, 144, '#34495e'), text(77, 149, 'r0', 14),
        marker(132, 144, '#34495e', repeat=1), text(147, 149, 'r1', 14),
        marker(227, 144, RED, failed=True), text(242, 149, 'Red X=failed drain', 14),
        marker(475, 144, PURPLE, diagnostic=True), text(490, 149, 'split: cache-confounded diagnostic', 14),
        marker(850, 144, GREY, missing=True), text(865, 149, 'Unavailable value (separate strip)', 14),
        text(50, 180, 'Horizontal offsets separate arms and repeats within each offered rate; points are categorical, not a fitted trend.', 14),
    ])
    for workload, left in PANEL_LEFT.items():
        rates = RATES[workload]
        spacing = PANEL_WIDTH / len(rates)
        centers = {rate: left + (index + 0.5) * spacing for index, rate in enumerate(rates)}
        panel_title = 'W1: eight turns per conversation' if workload == 'W1' else 'W2: four turns per conversation'
        parts.append(text(left, 209, panel_title, 18, **{'font-weight': 'bold'}))
        for tick in ticks:
            y = y_position(tick)
            parts.extend([line(left, y, left + PANEL_WIDTH, y),
                          text(left - 12, round(y + 5, 3), f'{tick:g}', 14, **{'text-anchor': 'end'})])
        parts.extend([line(left, TOP, left, BOTTOM, '#617181'),
                      line(left, BOTTOM, left + PANEL_WIDTH, BOTTOM, '#617181')])
        for rate, center in centers.items():
            parts.extend([line(center, BOTTOM, center, BOTTOM + 6, '#617181'),
                          text(round(center, 3), BOTTOM + 29, f'{rate:g}', 15, **{'text-anchor': 'middle'})])
        parts.extend([text(left + PANEL_WIDTH / 2, 611, 'Offered conversations per second', 15, **{'text-anchor': 'middle'}),
                      f'<rect x="{left}" y="636" width="{PANEL_WIDTH}" height="58" rx="4" fill="#f1f4f7"/>',
                      text(left + 8, 653, 'Unavailable (no numeric Y value)', 12, fill='#536577')])
        for row in rows:
            if row['workload'] != workload:
                continue
            value = row[key]
            diagnostic = row['disposition'] == 'cache_confounded_diagnostic'
            label, color, arm_offset = ARMS[row['arm']]
            x = centers[row['rate']] + arm_offset + (-6 if row['repeat'] == 0 else 6)
            if diagnostic:
                x -= 26
            y = 676 if value is None else y_position(value)
            exact_value = json.dumps(value, allow_nan=False)
            attributes = {'data-capture-id': row['capture_id'], 'data-value': exact_value,
                          'data-disposition': row['disposition'], 'data-arm': row['arm'],
                          'data-workload': workload, 'data-rate': row['rate'], 'data-repeat': row['repeat']}
            attrs = ' '.join(f'{name}="{escape(str(content), quote=True)}"' for name, content in attributes.items())
            if value is None:
                reason = ('NoSpec acceptance unavailable' if key == 'acceptance_length' and row['arm'] in ('D', 'B')
                          else 'native telemetry inconclusive' if key == 'acceptance_length'
                          else f'p95 unavailable: {row["successful_turns"]} successful turns; 20 required')
            else:
                reason = f'{ylabel}: {exact_value}'
            tooltip = f'{row["capture_id"]} | {row["disposition"]} | {reason}'
            parts.extend([f'<g {attrs}><title>{escape(tooltip)}</title>',
                          marker(x, y, GREY if value is None else color, row['repeat'],
                                 row['disposition'] == 'failed_drain', diagnostic, value is None), '</g>'])
            if diagnostic:
                parts.append(text(round(x - 10, 3), round(y - 12, 3), 'split', 12,
                                  fill=PURPLE, **{'text-anchor': 'end', 'font-weight': 'bold'}))
    parts.append(text(27, (TOP + BOTTOM) / 2, ylabel, 16,
                      transform=f'rotate(-90 27 {(TOP + BOTTOM) / 2:g})', **{'text-anchor': 'middle'}))
    if key == 'acceptance_length':
        notes = ['Acceptance length = 1 + accepted-token deltas / draft-step deltas; the Y axis starts above zero.',
                 'NoSpec acceptance is unavailable. A/W2/0.02 acceptance is unavailable because native telemetry is inconclusive.']
    else:
        notes = ['p95 uses successful turns only, including at failed rates; nearest-rank p95 requires at least 20 successful turns.',
                 'W2 at 0.02 has 16 successful turns per arm, so all four p95 values are unavailable. Turns within conversations are correlated.']
    notes.extend(['W1 and W2 have variable prompt lengths. The split diagnostic has unmatched cache state; maximum sustainable rates remain unestablished.',
                  'All numeric marks use the shared Y scale. Unavailable marks have no numeric value; horizontal position identifies offered rate and arm.'])
    parts.extend(text(50, 728 + index * 25, note, 14) for index, note in enumerate(notes))
    parts.append('</svg>')
    return ('\n'.join(parts) + '\n').encode('utf-8')


def expected_figures(root):
    summary = json.loads((Path(root) / 'summary.json').read_text())
    return {name: render(summary, name) for name in METRICS}


def verify_figures(root):
    """Compare both figures with deterministic output without writing files."""
    root = Path(root)
    for name, expected in expected_figures(root).items():
        if (root / 'figures' / name).read_bytes() != expected:
            raise ValueError('figure differs from summary and renderer: ' + name)
    return {'status': 'passed', 'figures': 2, 'captures_per_figure': 40}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        result = verify_figures(args.directory)
    else:
        expected = expected_figures(args.directory)
        for name, content in expected.items():
            (args.directory / 'figures' / name).write_bytes(content)
        result = verify_figures(args.directory)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
