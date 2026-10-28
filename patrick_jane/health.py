"""Audits and registered stop criteria; no policy actions or reward adjustments."""
from math import comb,isclose
from statistics import median


def audit_rollout(rollout):
    for match in rollout['matches']:
        if match['metrics']['illegal_actions'] or match.get('monitor_error'):
            raise RuntimeError('Legality or observer regression')
        if match['individual_decisions']!=match['neural_actions'].get('travel',0)+sum(match['neural_actions'].get(k,0) for k in ('task','report','repair','emergency','silent','share_evidence','accuse','defend','question','strategic_claim','skip','vote')):
            raise RuntimeError('Individual action accounting mismatch')
    for owner,components in rollout['reward_components'].items():
        index=int(owner.split('/')[0]);match=rollout['matches'][index]
        if components.get('exploration',0)>.3+1e-7 or components.get('new_player',0)>.7+1e-7:
            raise RuntimeError('Repeated novelty reward exploit')
        if components.get('task',0)+components.get('ghost_task',0)>4+1e-7:
            raise RuntimeError('Duplicated task reward exploit')
        if components.get('death',0)<-1-1e-7 or components.get('report',0)>1.1*match['crew_reports']+1e-7:
            raise RuntimeError('Duplicated death/report accounting')
        expected=10 if match['winner']=='crewmate' else -10 if match['winner']=='impostor' else 0
        if not isclose(components.get('win',0)+components.get('loss',0),expected,abs_tol=1e-7):
            raise RuntimeError('Terminal owner reward attribution mismatch')


def severe_paired_regression(reference,current):
    old={m['seed']:m['winner']=='crewmate' for m in reference}
    if set(old)!={m['seed'] for m in current}:raise ValueError('Regression check requires paired scenarios')
    lost=sum(old[m['seed']] and m['winner']!='crewmate' for m in current)
    gained=sum(not old[m['seed']] and m['winner']=='crewmate' for m in current)
    n=lost+gained
    p=sum(comb(n,k) for k in range(gained+1))/2**n if n and lost>gained else 1.
    return dict(stop=(lost-gained)/len(current)>=.25 and p<.01,lost=lost,gained=gained,one_sided_p=p)


def severe_speed_regression(history):
    if len(history)<6:return False
    rates=[r['rollout']['decisions_per_second'] for r in history]
    return all(r<.4*median(rates[:3]) for r in rates[-3:])
