"""Standalone research figures; training win rate is never labelled accuracy."""
import argparse
import json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args(argv)
    root=args.run;out=root/'graphs';out.mkdir(exist_ok=True)
    registration=json.loads((root/'registration.json').read_text())
    rows=[json.loads(line) for line in (root/'matches.jsonl').read_text().splitlines() if line.strip()]
    updates=json.loads((root/'ppo.json').read_text()) if (root/'ppo.json').exists() else []
    label=f"Lizbon · seed {registration['initialization_seed']} · {registration['mode']}"
    x=np.arange(1,len(rows)+1)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    completed=[r for r in rows if not r['truncated'] and r['winner']]
    counts=Counter(r['reason'] for r in rows)
    axes[0,0].bar(list(counts),list(counts.values()),color='#3c7ca8');axes[0,0].set_title('Outcomes and technical cuts')
    axes[0,0].tick_params(axis='x',rotation=20)
    returns=[np.mean([sum(v.values()) for v in r['impostor_rewards'].values()]) for r in rows]
    axes[0,1].plot(x,returns,'o-');axes[0,1].set_title('Mean impostor return per match')
    axes[1,0].plot(x,[r['metrics']['eliminations'] for r in rows],'o-');axes[1,0].set_title('Accepted kills per match')
    if completed:
        wins=np.cumsum([r['winner']=='impostor' for r in completed])/np.arange(1,len(completed)+1)
        axes[1,1].plot(np.arange(1,len(wins)+1),wins,'o-');axes[1,1].set_ylim(0,1)
    else:axes[1,1].text(.5,.5,'No natural completed games\nNo win-rate estimate',ha='center',va='center')
    axes[1,1].set_title('Cumulative outcome rate; not test accuracy')
    fig.suptitle(label+' · observed outcomes');fig.savefig(out/'outcomes.png',dpi=160);plt.close(fig)
    actions=Counter()
    for r in rows:
        for actor in r['impostor_actions'].values():
            actions.update({k:v for k,v in actor.items() if k not in ('observations','choices','forced_waits')})
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');items=actions.most_common()
    ax.barh([k for k,v in items][::-1],[v for k,v in items][::-1],color='#397d6c')
    ax.set_xlabel('Sampled requests; accepted effects recorded separately');ax.set_title(label+' · action distribution')
    fig.savefig(out/'actions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,key in zip(axes.flat,('policy','value','entropy','kl')):
        if updates:ax.plot([r[key] for r in updates])
        else:ax.text(.5,.5,'No PPO updates',ha='center',va='center')
        ax.set_title(key);ax.set_xlabel('Optimizer pass')
    fig.suptitle(label+' · PPO diagnostics');fig.savefig(out/'ppo.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(x,[r['wall_seconds'] for r in rows],label='collection wall s')
    axes[0].plot(x,[r.get('optimization_seconds',0) for r in rows],label='optimization wall s');axes[0].legend()
    axes[1].plot(x,[r['impostor_observations']/max(.001,r['wall_seconds']) for r in rows],'o-')
    axes[1].set_title('Impostor observations / collection wall second')
    fig.suptitle(label+' · timing');fig.savefig(out/'timing.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(graphs=[str(p) for p in sorted(out.glob('*.png'))],natural_games=len(completed))))


if __name__=='__main__':main()
