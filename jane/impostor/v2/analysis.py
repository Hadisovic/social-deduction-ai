"""Offline accounting of all neural choices; no claims about hidden intentions."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path


def summarize_decisions(path):
    actors={};last={};effects=Counter();failures=[]
    with gzip.open(path,'rt',encoding='utf-8') as handle:
        for line in handle:
            r=json.loads(line)
            if r['record']=='decision':
                pid=r['owner'];a=actors.setdefault(pid,Counter());prev=last.get(pid)
                a['decisions']+=1;a['selected/'+r['selected_kind']]+=1
                a['forced_single_candidate' if len(r['candidates'])==1 else 'voluntary_choice']+=1
                eligible={c['kind'] for c in r['candidates']}
                continuous=prev is not None and prev['phase']==r['phase'] and 0<r['time']-prev['time']<=.600001
                for kind in ('kill','report','fake','lights','doors','vent_enter','vote','claim'):
                    if kind in eligible:
                        a[kind+'_eligible_decisions']+=1
                        if not continuous or kind not in prev['eligible']:a[kind+'_windows']+=1
                selected=r['candidates'][r['selected']]
                request=(selected['kind'],selected['key'])
                streak=prev['streak']+1 if continuous and request==prev['request'] else 1
                a['longest_identical_request_streak']=max(a['longest_identical_request_streak'],streak)
                local=r['local']
                if r['phase']=='roaming' and r['selected_kind']=='wait' and len(r['candidates'])>1:
                    a['voluntary_continue_or_wait']+=1
                    if local.get('navigation')!='moving' and not local.get('door_blocked') and not local.get('vent') and not local.get('fake_remaining',0):
                        a['idle_wait_decisions']+=1
                if r['selected_kind']=='kill':a['selected_kill_probability_sum']+=selected['probability']
                if r['selected_kind'] in ('destination','room','visible_player','remembered_player','body_travel'):
                    a['goal/'+selected['key']]+=1
                a['entropy_sum']+=r.get('entropy',0.)
                last[pid]=dict(eligible=eligible,time=r['time'],phase=r['phase'],request=request,streak=streak)
            elif r['record']=='effects':
                for e in r['events']:effects[e['kind']]+=1
                if r['phase']=='roaming':
                    for pid,p in r['impostors'].items():
                        if not p['active']:continue
                        a=actors.setdefault(pid,Counter());dt=r['elapsed']
                        a['alive_roaming_seconds']+=dt
                        moving=math.hypot(*p['velocity'])>1e-6
                        a['moving_seconds' if moving else 'stationary_seconds']+=dt
                        if p.get('fake','') and p['fake'].startswith('fake:'):a['fake_seconds']+=dt
                        if p['cooldown']<=1e-9 and not p['vented']:
                            a['ready_moving_seconds' if moving else 'ready_stationary_seconds']+=dt
            elif r['record']=='reward':
                a=actors.setdefault(r['owner'],Counter())
                for k,v in r['components'].items():a['reward/'+k]+=v
            elif r['record']=='failure':failures.append(r)
    result={}
    for pid,a in actors.items():
        if not a['decisions']:continue  # Native recorder also delivers frozen-crew rewards.
        result[pid]={k:v for k,v in a.items() if not k.startswith('goal/')}
        result[pid]['top_goals']=Counter({k[5:]:v for k,v in a.items() if k.startswith('goal/')}).most_common(5)
        result[pid]['mean_entropy']=a['entropy_sum']/max(1,a['decisions'])
    return dict(actors=result,effects=dict(effects),failures=failures,
        interpretation='Policy choices and observed effects, not an explanation of internal neural intentions',
        caveats=['Windows are contiguous availability intervals, not independent trials.',
            'Stationary includes legal waiting/obstruction; inspect replay before calling it camping.',
            'Ready stationary uses tick endpoints; authoritative ready-time rewards use boundary accounting.',
            'Offline world effects are diagnostic only; actor input is logged separately.'])


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args()
    rows={p.name:summarize_decisions(p) for p in sorted(args.run.glob('*.decisions.jsonl.gz'))}
    (args.run/'decision-analysis.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(matches=len(rows),output=str(args.run/'decision-analysis.json'))))


if __name__=='__main__':main()
