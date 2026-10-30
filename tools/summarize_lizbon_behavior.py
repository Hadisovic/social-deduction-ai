"""Read finalized public-study replays; diagnostic truth never enters the actor."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def summarize(path):
    actors={};previous_time=0.;roles={};last_request={};streak={};last_tick={}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            r=json.loads(line)
            if r['record']=='header':
                roles=r['agent_roles'];actors={p:Counter() for p,v in roles.items() if v=='impostor'}
            elif r['record']=='neural' and r['player_id'] in actors:
                pid=r['player_id'];a=actors[pid];kind=r['action_kind'];target=r.get('action_target','')
                a['decisions']+=1;a['selected/'+kind]+=1
                for ability in ('kill','report','fake'):
                    eligible=any(k.split(':',2)[1:2]==[ability] and v>0 for k,v in r.get('action_probabilities',{}).items())
                    a[ability+'_opportunity_decisions']+=int(eligible)
                    if eligible and kind==ability:a[ability+'_chosen_when_available']+=1
                request=(kind,target)
                streak[pid]=streak.get(pid,0)+1 if last_request.get(pid)==request else 1
                last_request[pid]=request
                a['longest_identical_request_streak']=max(a['longest_identical_request_streak'],streak[pid])
                if kind in ('destination','room','visible_player','remembered_player','body_travel'):
                    a['goal/'+str(target)]+=1
            elif r['record']=='tick':
                now=float(r['sim_time']);elapsed=max(0.,now-previous_time);previous_time=now
                if r['phase']!='roaming':continue
                for pid,a in actors.items():
                    p=r['players'][pid]
                    if p['status']!='alive':continue
                    a['alive_roaming_seconds']+=elapsed
                    # Endpoint velocity approximates motion over this replay tick.
                    moving=math.hypot(*p['velocity'])>1e-6
                    a['moving_seconds' if moving else 'stationary_seconds']+=elapsed
                    if p.get('interaction_task','') and p['interaction_task'].startswith('fake:'):
                        a['fake_interaction_seconds']+=elapsed
                    if float(p.get('cooldown',1))<=1e-9:
                        a['ready_moving_seconds' if moving else 'ready_stationary_seconds']+=elapsed
    clean={}
    for pid,counts in actors.items():
        goals={k[5:]:v for k,v in counts.items() if k.startswith('goal/')}
        clean[pid]={k:v for k,v in counts.items() if not k.startswith('goal/')}
        clean[pid]['top_goal_requests']=Counter(goals).most_common(5)
        clean[pid]['distinct_requested_goals']=len(goals)
    return dict(replay=path.name,actors=clean)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();root=args.run
    registration=json.loads((root/'registration.json').read_text())
    if registration['mode'] not in ('train','evaluate','smoke'):raise ValueError('Unrecognized study mode')
    rows=[summarize(p) for p in sorted(root.glob('match-*.jsonl'))]
    report=dict(method='Offline spectator replay diagnostics; no privileged actor input',matches=rows,
        caveats=['Stationary includes genuine waiting, vent time and door obstruction; it is not proof of camping.',
                 'Ready-stationary is descriptive; the reward ledger separately excludes ineligible intervals.',
                 'Repeated target requests may be legitimate revisits or updated visible targets.',
                 'Opportunity fractions count actor decisions, not independent kill opportunities.',
                 'Motion/fake durations are endpoint tick approximations; meetings and eliminated actors excluded.'])
    (root/'behavior.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    labels=[];metrics=[]
    for i,row in enumerate(rows,1):
        for pid,values in row['actors'].items():labels.append(f'{i}/{pid}');metrics.append(values)
    for key in ('moving_seconds','stationary_seconds','fake_interaction_seconds'):
        axes[0].plot(labels,[r.get(key,0) for r in metrics],'o-',label=key.replace('_',' '))
    axes[0].set_ylabel('Simulated seconds; fake overlaps stationary');axes[0].legend(fontsize=8)
    axes[0].tick_params(axis='x',rotation=45);axes[0].set_title('Roaming behavior; stationary is not necessarily camping')
    for ability in ('kill','report','fake'):
        available=sum(r.get(ability+'_opportunity_decisions',0) for r in metrics)
        chosen=sum(r.get(ability+'_chosen_when_available',0) for r in metrics)
        axes[1].bar(ability,chosen/available if available else 0)
        axes[1].text(ability,(chosen/available if available else 0)+.03,f'{chosen}/{available}',ha='center')
    axes[1].set_ylim(0,1.15);axes[1].set_ylabel('Selected / eligible actor decisions')
    axes[1].set_title('Action availability responses; counts are dependent')
    fig.suptitle(f"Lizbon · {registration['mode']} · offline behavior")
    folder=root/'graphs';folder.mkdir(exist_ok=True);fig.savefig(folder/'behavior.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(matches=len(rows),output=str(root/'behavior.json'))))


if __name__=='__main__':main()
