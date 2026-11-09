"""Static publication figures from the exported analysis tables."""
import csv
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT/'data', ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold',
    'savefig.facecolor':'white','svg.fonttype':'none'})
NS, K5 = '#3266a8', '#c45d1e'


def csv_rows(name):
    with (DATA/name).open() as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    svg=FIG/(name+'.svg')
    fig.savefig(svg, bbox_inches='tight')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(FIG/(name+'.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    blocks = csv_rows('natural-blocks.csv')
    regimes = {r['regime']:r for r in csv_rows('natural-regimes.csv')}
    keys = ['8000-c1','8000-c8','64000-c1','64000-c8']
    fig, ax = plt.subplots(figsize=(8.7,4.7))
    for i, key in enumerate(keys):
        rows = [r for r in blocks if r['regime']==key]
        ax.scatter([float(r['token_throughput_ratio']) for r in rows],
            [i+j*.035-.07 for j in range(len(rows))],color='#79808b',s=34,zorder=3)
        r=regimes[key];lo,hi,med=(float(r[x]) for x in ['interval_low','interval_high','median_ratio'])
        ax.plot([lo,hi],[i,i],color=K5,lw=3,alpha=.7)
        ax.scatter([med],[i],marker='D',color=K5,s=65,zorder=4,
            label='Median and descriptive 95% block-bootstrap interval' if i==0 else None)
    ax.axvline(1,color='#444',ls='--',lw=1)
    ax.set_xscale('log');ax.set_xticks([.5,1,2,4,8,16]);ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks(range(4),['~8K input / c1','~8K input / c8','~64K input / c1','~64K input / c8']);ax.invert_yaxis()
    ax.set_xlabel('K5 / NoSpec output-token throughput ratio (log scale)')
    ax.set_title('Natural-output token throughput varies by prompt block',loc='left',pad=16)
    ax.grid(axis='x',alpha=.15);ax.legend(loc='lower right',fontsize=8)
    fig.text(.01,-.03,'Five prompt blocks per regime; shared engine runs, differing answer lengths and correctness.\nGray points show every block. Ratios count all valid output tokens, including incorrect answers.',fontsize=9)
    save(fig,'natural-throughput')

    windows=csv_rows('load-windows.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),sharey=True)
    for ax,point,title in zip(axes,['mixed-low','mixed-high'],['1.5 offered requests/s','6 offered requests/s']):
        for arm,color in [('nospec',NS),('k5',K5)]:
            rows=[r for r in windows if r['profile']=='fast' and r['arm']==arm and r['window'].startswith(point)]
            rows.sort(key=lambda r:r['window'])
            ax.plot([1,2],[1000*float(r['all_recorded_first_visible_p95']) for r in rows],
                marker='o',color=color,label='NoSpec' if arm=='nospec' else 'Static K5')
        ax.set_xticks([1,2],['Repetition 1','Repetition 2']);ax.set_title(title,loc='left');ax.grid(axis='y',alpha=.2)
    axes[0].set_yscale('log');axes[0].set_ylabel('Fast first-visible p95, milliseconds (log scale)');axes[1].legend()
    fig.suptitle('K5 has lower recorded Fast first-visible latency in these windows',x=.01,ha='left',weight='bold',y=1.05)
    fig.text(.01,-.08,'All dispatched Fast requests with visible-output timestamps, including incorrect answers and straddlers.\nEach window is 60 seconds; repeated tasks and shared engine runs limit generalization.',fontsize=9)
    save(fig,'first-visible')

    fig,axes=plt.subplots(1,2,figsize=(10,4.1),sharey=True)
    deadlines=csv_rows('load-deadlines.csv')
    for i,ax in enumerate(axes):
        window='mixed-high-r'+str(i+1)
        for arm,color,shift in [('nospec',NS,-.17),('k5',K5,.17)]:
            w=next(r for r in windows if r['profile']=='all' and r['window']==window and r['arm']==arm)
            d=next(r for r in deadlines if r['profile']=='all' and r['window']==window and r['arm']==arm and r['deadline_seconds']=='1')
            values=[float(w['completed_per_second']),float(w['successful_per_second']),float(d['correct_within_deadline_per_second'])]
            ax.bar([j+shift for j in range(3)],values,width=.32,color=color,label='NoSpec' if arm=='nospec' else 'Static K5')
            for j,v in enumerate(values):ax.text(j+shift,v+.055,f'{v:.2f}',ha='center',fontsize=8)
        ax.axhline(6,color='#444',ls='--',lw=1);ax.set_ylim(0,6.8);ax.set_xticks([0,1,2],['Completed','Correct','Correct ≤1 s']);ax.set_title('Repetition '+str(i+1),loc='left')
    axes[0].set_ylabel('Requests per second within the 60-second window');axes[1].legend(loc='upper right',framealpha=1)
    fig.suptitle('Fixed offered traffic limits total throughput; a deadline reveals latency value',x=.01,ha='left',weight='bold',y=1.05)
    fig.text(.01,-.08,'Dashed line: 6 offered requests/s (Fast 4/s + Think 2/s). Correct ≤1 s also requires completion\ninside the measurement window. One second is a predeclared sensitivity limit, not a production SLA.',fontsize=9)
    save(fig,'task-rates')

    rows=csv_rows('quality.csv');fig,ax=plt.subplots(figsize=(8.7,3.7))
    for i,r in enumerate(rows):
        diff=float(r['difference_pp']);lo=float(r['interval_low_pp']);hi=float(r['interval_high_pp'])
        ax.errorbar(diff,i,xerr=[[diff-lo],[hi-diff]],fmt='D',color=K5,capsize=5,markersize=7)
        ax.text(8.8,i,f"{r['tasks']} tasks × 2 seeds",va='center',fontsize=9)
    ax.axvline(0,color='#444',ls='--');ax.set_xlim(-4,13);ax.set_yticks(range(len(rows)),[r['profile'].capitalize() for r in rows]);ax.invert_yaxis()
    ax.set_xlabel('K5 minus NoSpec task success, percentage points');ax.set_title('Selected-panel quality differences are small and uncertain',loc='left',pad=16)
    fig.text(.01,-.15,'Nominal task-ID bootstrap intervals, conditional on this panel. Tool-template dependence and only\n20 Think tasks limit interpretation; this panel does not establish general quality equivalence.',fontsize=9)
    save(fig,'task-quality')
    print('Wrote four SVG/PNG figures with all repeated observations and explicit metric populations.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=FIG,help='Figure output directory; use scratch space to preserve sealed exports')
    args=parser.parse_args()
    FIG=args.output.resolve()
    FIG.mkdir(parents=True,exist_ok=True)
    main()
