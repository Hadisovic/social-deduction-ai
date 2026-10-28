"""Read-only completed-recording, eligibility and frozen-meeting audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--replays',required=True,type=Path);p.add_argument('--spectator-source',required=True,type=Path)
    a=p.parse_args();sys.path.append(str(a.spectator_source.resolve()))
    from spectator.replay_format import parse_replay
    totals=Counter();tasks=Counter();failures=[]
    for path in sorted(a.replays.rglob('*.jsonl')):
        parsed=parse_replay(path);assert parsed['complete'],path.name
        current=None;previous=None;eligible={};end=None;header=None
        for line in path.open(encoding='utf-8'):
            e=json.loads(line)
            if e['record']=='header':header=e
            if e['record']=='tick':
                previous=current;current=e
                if previous and previous['phase'] in ('discussion','voting') and current['phase'] in ('discussion','voting'):
                    for pid,pdata in current['players'].items():
                        old=previous['players'][pid]
                        for key in ('position','ghost_position','tasks_completed','task_progress'):
                            if pdata.get(key)!=old.get(key):failures.append(dict(file=path.name,check='meeting freeze',key=key))
                    totals['meeting_adjacent_tick_pairs']+=1
            if e['record']=='event':
                if e['kind']=='meeting_started':eligible[e['meeting_id']]=set(e['participants'])
                if e['kind']=='vote_submitted':
                    assert e['voter_id'] in eligible[e['meeting_id']]
                    totals['eligible_submitted_ballots']+=1
                if e['kind']=='illegal_action':failures.append(dict(file=path.name,check='illegal action'))
                if e['kind']=='task_completed' and float(e['time'])>=header['config'].get('prefix_handoff_seconds',0):tasks[header['config']['opponent']]+=1
            if e['record']=='neural':
                assert header['controller_types'][e['player_id']]=='ppo'
                assert current['players'][e['player_id']]['status']=='alive'
                totals['owner_bound_living_neural_decisions']+=1
            if e['record']=='result':end=e
        assert end and end['reason'] in ('PARITY','IMPOSTOR_EJECTED','TIMEOUT','TASKS_COMPLETED')
        assert end.get('metrics',{}).get('illegal_actions',0)==0
        totals['completed_recordings']+=1
    report=dict(status='passed' if not failures else 'failed',counts=dict(totals),task_events_after_handoff=dict(tasks),failures=failures,
        scope='Read-only actual new recordings; no general UI verification or gameplay execution.')
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    if failures:raise SystemExit(2)


if __name__=='__main__':main()
