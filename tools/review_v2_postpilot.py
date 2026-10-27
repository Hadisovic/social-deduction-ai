"""Offline audit of completed sampled pilot replays; no simulator executions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from among_us_map import get_map
from social_deduction.phase3_observation import visible_between


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replays',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    results={};world=get_map()
    for arm in ('original','shaped'):
        counts=Counter();last=Counter();ages=[];death_times=[]
        for f in (a.replays/arm).glob('*.jsonl'):
            rows=[json.loads(x) for x in f.open(encoding='utf-8')]
            header=rows[0];focal=header['config']['focal_player'];counts['completed_sampled_replays']+=1
            dead=set();phase='roaming'
            for row in rows:
                if row.get('record')!='event':continue
                kind=row.get('kind')
                if kind=='elimination':dead.add(row['victim_id'])
                if kind=='ejection' and row.get('player_id'):dead.add(row['player_id'])
                if kind=='meeting_started':phase='discussion'
                if kind=='voting_started':phase='voting'
                if kind=='exploration_resumed':phase='roaming'
                if kind=='vote_submitted':
                    counts['submitted_ballots']+=1;counts['dead_submitted_ballots']+=row['voter_id'] in dead
                if kind=='task_completed' and phase in ('discussion','voting'):
                    counts['meeting_task_completions']+=1
                    counts['meeting_ghost_task_completions']+=row['player_id'] in dead
            kills=[r for r in rows if r.get('record')=='event' and r.get('kind')=='elimination']
            counts['focal_first_killed']+=bool(kills and kills[0]['victim_id']==focal)
            death=next((r for r in kills if r['victim_id']==focal),None)
            if not death:continue
            counts['focal_killed']+=1;death_times.append(death['time'])
            neural=[r for r in rows if r.get('record')=='neural' and r['player_id']==focal and r['tick']<=death['tick']]
            if neural:
                n=neural[-1];label=n['action_kind'];last[label.split(': ',1)[-1].split(' ')[0]]+=1
                ages.append(death['time']-n['sim_time'])
                counts['last_choice_followed_eventual_killer']+='follow '+death['killer_id'] in label
            ticks=[r for r in rows if r.get('record')=='tick' and r['tick']<death['tick']]
            if ticks:
                tick=ticks[-1];own=tick['players'][focal]['position']
                nearby=[pid for pid,state in tick['players'].items() if pid!=focal and state['status']=='alive'
                        and header['agent_roles'][pid]=='crewmate' and visible_between(world,tuple(own),tuple(state['position']),4.5)]
                counts['prekill_snapshots_scored']+=1;counts['no_other_crew_visible_at_preceding_snapshot']+=not nearby
        results[arm]=dict(counts,last_choice_before_kill=dict(last),
            decision_to_death_seconds=dict(median=statistics.median(ages) if ages else None,maximum=max(ages) if ages else None),
            median_killed_at_seconds=statistics.median(death_times) if death_times else None)
    record=dict(scope='Offline, completed preselected training streams only; paired arms are not independent samples.',
                limitations='Preceding tick snapshots are approximate exposure, not causal safety labels. No validation replay census; no numeric historical suspicion trace.',arms=results)
    with a.output.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
