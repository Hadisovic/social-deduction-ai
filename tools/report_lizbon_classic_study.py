"""Offline pilot report, all-choice diagnostics and graphs; never runs a match."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def read(path,default=None):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);args=p.parse_args();root=args.run
    from jane.impostor.v2.analysis import summarize_decisions
    aggregate={};details={}
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and (p/'registration.json').exists()):
        rows=[json.loads(line) for line in (folder/'matches.jsonl').read_text().splitlines()] if (folder/'matches.jsonl').exists() else []
        choices={}
        for path in sorted(folder.glob('*.decisions.jsonl.gz')):
            try:choices[path.name]=summarize_decisions(path)
            except (EOFError,OSError,json.JSONDecodeError) as e:choices[path.name]=dict(partial_file_error=str(e))
        # Additional truth is offline only: attribute actual self-reports and witnessed kills.
        actual=Counter();killers={}
        for path in sorted(folder.glob('*.decisions.jsonl.gz')):
            try:
                with gzip.open(path,'rt',encoding='utf-8') as f:
                    for line in f:
                        row=json.loads(line)
                        if row['record']!='effects':continue
                        for event in row['events']:
                            kind=event['kind']
                            if kind=='elimination':killers[event['victim_id']]=event['killer_id']
                            if kind=='report':
                                reporter=event['reporter_id'];victim=event.get('victim_id',event.get('body_id'))
                                actual['report/'+reporter]+=1
                                if killers.get(victim)==reporter:actual['actual_self_report/'+reporter]+=1
                            if kind=='elimination_witnessed':actual['witnessed/'+event['killer_id']]+=1
                            if kind in ('a2_action_rejected','illegal_action','team_resource_contention','vote','structured_claim','claim'):
                                actual[kind]+=1
            except (EOFError,OSError,json.JSONDecodeError):pass
        details[folder.name]=dict(matches=choices,actual_effect_counts=dict(actual))
        natural=[r for r in rows if r['winner'] and not r['truncated']]
        aggregate[folder.name]=dict(matches=[dict(seed=r['seed'],winner=r['winner'],reason=r['reason'],truncated=r['truncated'],
            simulation_seconds=r['simulation_seconds'],wall_seconds=r['wall_seconds'],
            kills=r['metrics']['eliminations'],illegal=r['metrics']['illegal_actions'],
            navigation_failures=r['metrics']['navigation_failures'],impostor_actions=r['impostor_actions'],
            rewards=r['impostor_rewards'],optimization_seconds=r.get('optimization_seconds',0)) for r in rows],
            games=len(natural),wins=sum(r['winner']=='impostor' for r in natural),cuts=len(rows)-len(natural),
            loss_causes=dict(Counter(r['reason'] for r in natural if r['winner']=='crewmate')),
            counters=read(folder/'results.json',{}).get('counters',{}),failure=read(folder/'failure.json'),
            runtime=read(folder/'registration.json')['runtime'])
    (root/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n',encoding='utf-8')
    (root/'decision-analysis.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    graphs=root/'graphs';graphs.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,data in aggregate.items():
        if not name.startswith('train'):continue
        rows=data['matches'];x=list(range(1,len(rows)+1))
        axes[0].plot(x,[r['kills'] for r in rows],'o-',label=name)
        axes[1].plot(x,[r['rewards'] and sum(sum(v.values()) for v in r['rewards'].values())/len(r['rewards']) for r in rows],'o-',label=name)
    axes[0].set(title='Training matches: kills',xlabel='Match',ylabel='Kills; cuts remain descriptive')
    axes[1].set(title='Training team rewards',xlabel='Match',ylabel='Mean return per impostor')
    for ax in axes:ax.legend(fontsize=8)
    fig.suptitle('Training outcomes are not test accuracy');fig.savefig(graphs/'training.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for folder in root.glob('train*'):
        if not folder.is_dir():continue
        logs=read(folder/'ppo.json',[])
        for ax,key in zip(axes,('entropy','value','kl')):
            ax.plot([r[key] for r in logs],label=folder.name);ax.set(title=key,xlabel='Completed optimizer pass')
    for ax in axes:ax.legend(fontsize=8)
    fig.suptitle('Recurrent PPO diagnostics; sparse pilot data');fig.savefig(graphs/'ppo.png',dpi=150);plt.close(fig)
    labels=[n for n in aggregate if n.startswith(('validation','confirmation'))]
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    for i,name in enumerate(labels):
        d=aggregate[name];n=d['games'];w=d['wins'];rate=w/n if n else 0
        # Wilson 95% interval; small n remains visibly uncertain.
        if n:
            z=1.96;den=1+z*z/n;center=(rate+z*z/(2*n))/den
            radius=z*((rate*(1-rate)/n+z*z/(4*n*n))**.5)/den
            ax.errorbar(i,rate,yerr=[[max(0.,rate-(center-radius))],[max(0.,(center+radius)-rate)]],fmt='o',capsize=4)
        ax.text(i,rate+.08,f'{w}/{n}\ncuts {d["cuts"]}',ha='center',fontsize=8)
    ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right');ax.set_ylim(-.1,1.25)
    ax.set(title='Development win fractions and Wilson 95% intervals',ylabel='Natural wins / completed matches')
    fig.savefig(graphs/'evaluation.png',dpi=150);plt.close(fig)
    counts=Counter();moving=stationary=fake=0.
    for data in details.values():
        for match in data['matches'].values():
            for actor in match.get('actors',{}).values():
                counts.update({k[9:]:v for k,v in actor.items() if k.startswith('selected/')})
                moving+=actor.get('moving_seconds',0);stationary+=actor.get('stationary_seconds',0);fake+=actor.get('fake_seconds',0)
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    kinds=counts.most_common();axes[0].bar([k for k,v in kinds],[v for k,v in kinds]);axes[0].tick_params(axis='x',rotation=60)
    axes[0].set(title='All logged impostor selections',ylabel='Decisions; correlated samples')
    axes[1].bar(['moving','stationary','fake'],[moving,stationary,fake])
    axes[1].set(title='Roaming behavior; fake overlaps stationary',ylabel='Actor simulated seconds')
    fig.savefig(graphs/'behavior.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
    names=list(aggregate);collection=[sum(r['wall_seconds'] for r in aggregate[n]['matches'])/60 for n in names]
    optimization=[sum(r['optimization_seconds'] for r in aggregate[n]['matches'])/60 for n in names]
    ax.bar(names,collection,label='collection');ax.bar(names,optimization,bottom=collection,label='PPO')
    ax.tick_params(axis='x',rotation=40);ax.set(title='Measured work per stream/method; two workers overlap',ylabel='Worker wall minutes');ax.legend()
    fig.savefig(graphs/'timing.png',dpi=150);plt.close(fig)
    status=read(root/'status.json',{});reg=read(root/'registration.json',{});lock=read(root/'selection-lock.json')
    lines=['# Lizbon Classic V2 pilot report','',f"Status: **{status.get('phase','unknown')}**.",
        f"Started: {reg.get('started_utc')}. Hard deadline: {reg.get('deadline_utc')}.",
        f"Actual supervisor wall time: {status.get('wall_seconds',0)/60:.2f} minutes.",
        f"Source: `{reg.get('source_commit')}`. Two same-start sampling streams; not two independent initializations.",'',
        '| Run | Wins / natural games | Cuts | Kills | Loss causes |','|---|---:|---:|---:|---|']
    for name,d in aggregate.items():lines.append(f"| {name} | {d['wins']}/{d['games']} | {d['cuts']} | {sum(r['kills'] for r in d['matches'])} | {d['loss_causes']} |")
    lines+=['',f"Locked selection: {lock['selected'] if lock else 'none; confirmation must remain unopened'}.",
        '', '## Learning sufficiency','',
        'This small pilot cannot establish reliable searching, deception, population generalization or a globally best policy. '
        'Training wins are not accuracy. Check the paired confirmation, behavior and uncertainty before claiming improvement. '
        'The frozen crew is learned task-pursuit update102, not an A2-trained strategic crew. '
        'Only two confirmation cases per inference mode are registered; conclusions remain preliminary.',
        '', '## Graphs','',
        '![Training](graphs/training.png)','Kills and return across training games; inspect cuts separately.',
        '![PPO](graphs/ppo.png)','Entropy measures policy spread; value loss measures return fit; KL estimates policy movement. None is win accuracy.',
        '![Evaluation](graphs/evaluation.png)','Development comparisons with very wide small-sample intervals. No reserved final set was opened.',
        '![Behavior](graphs/behavior.png)','All choices and motion. Stationary can include deliberate waiting/door obstruction; faking overlaps it.',
        '![Timing](graphs/timing.png)','Per-worker collection/PPO time. Parallel worker times must not be added to claim elapsed study time.',
        '', '## What to inspect next','',
        'Use decision-analysis.json for each actor: repeat streaks, kill/report availability windows, chosen actions, '
        'ready-time idling, faking, self-reports, witnessed kills and rewards. The raw compressed audits retain every '
        'candidate probability and command. Replay context is required before diagnosing a strategy.',
        '', 'Keep frozen cross-play first. The impostor developer should test searching and kill selection against varied '
        'frozen crews. The crew developer should validate reporting and witnessed-kill reasoning on the same rules/profile. '
        'Add population/task curricula only after fixed-profile competence; adaptation and self-play come later.']
    if (root/'failure.json').exists():lines+=['','## Preserved failure','',json.dumps(read(root/'failure.json'),indent=2)]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(root/'REPORT.md'),runs=len(aggregate))))


if __name__=='__main__':main()
