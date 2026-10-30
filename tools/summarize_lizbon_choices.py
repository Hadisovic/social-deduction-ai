"""Offline action/effect accounting, including recoverable interrupted prefixes."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def audit(path):
    rows=[];read_error=None
    try:
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:read_error='Incomplete JSON line';break
    except (EOFError,OSError) as e:read_error=str(e)
    actors={};effects=Counter();killers={};witnesses={};last=None
    for r in rows:
        if r['record']=='decision':
            pid=r['owner'];a=actors.setdefault(pid,Counter());c=r['candidates'][r['selected']];local=r['local']
            a['decisions']+=1;a['selected/'+c['kind']]+=1
            a['forced_waits' if len(r['candidates'])==1 else 'voluntary_choices']+=1
            available={x['kind'] for x in r['candidates']}
            for kind in ('kill','report','fake','claim','vote','lights','doors','vent_enter'):
                if kind in available:a['eligible_decisions/'+kind]+=1
            if c['kind'] in ('visible_player','remembered_player'):
                a['travel_to_ally' if c['key'] in local['teammates'] else 'travel_to_nonally']+=1
            if c['kind']=='kill':
                bystanders=set(local['visible_players'])-set(local['teammates'])-{c['key']}
                a['kill_choices_with_visible_crew_bystanders']+=bool(bystanders)
            if c['kind']=='vote':a['votes_for_ally']+=c['key'] in local['teammates']
            if c['kind']=='claim':
                claim=c['command'].get('claim');statement=c['command'].get('statement')
                kind=claim['kind'] if claim else statement.get('kind','other') if statement else 'other'
                a['claim_type/'+kind]+=1
            last=dict(tick=r['tick'],time=r['time'],phase=r['phase'],owner=pid,kind=c['kind'],target=c['key'])
        elif r['record']=='effects':
            for e in r['events']:
                kind=e['kind'];effects[kind]+=1
                if kind=='elimination':killers[e['victim_id']]=e['killer_id']
                if kind=='elimination_witnessed':witnesses.setdefault(e['victim_id'],set()).add(e['observer_id'])
                if kind=='report' and killers.get(e['victim_id'])==e['reporter_id']:
                    actors.setdefault(e['reporter_id'],Counter())['actual_self_reports']+=1
    imp_ids=set(actors)
    for victim,killer in killers.items():
        crew_witnesses=witnesses.get(victim,set())-imp_ids-{victim}
        actors.setdefault(killer,Counter())['accepted_kills']+=1
        actors[killer]['kills_with_actual_other_living_crew_witness']+=bool(crew_witnesses)
    return dict(decisions=sum(a['decisions'] for a in actors.values()),actors={p:dict(a) for p,a in actors.items()},
        effects=dict(effects),last_logged_choice=last,read_error=read_error,
        caveats=['Counts describe choices/effects; they do not reveal neural intentions.',
            'Visible bystanders are actor-local possibilities; actual witnesses are separate offline truth.',
            'Eligibility counts are correlated decisions, not independent trials or accuracy.',
            'Interrupted prefixes have no fabricated terminal result; partial gzip recovery retains readable rows only.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);args=p.parse_args()
    report={str(path.relative_to(args.run)):audit(path) for path in sorted(args.run.glob('train*/*.decisions.jsonl.gz'))}
    (args.run/'choice-effects.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(matches=len(report),decisions=sum(v['decisions'] for v in report.values()))))


if __name__=='__main__':main()
