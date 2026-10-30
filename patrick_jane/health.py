"""Audits and registered stop criteria; no policy actions or reward adjustments."""
from math import comb,isclose
from statistics import median
from .rewards import COEFFICIENTS as C


def audit_rollout(rollout):
    for match in rollout['matches']:
        if match['metrics']['illegal_actions'] or (match.get('monitor_error') and match.get('observer_errors_fatal',True)):
            raise RuntimeError(f"Seed {match.get('seed')}: illegal_actions={match['metrics']['illegal_actions']}, monitor_error={match.get('monitor_error')}")
        if match['individual_decisions']!=match['neural_actions'].get('travel',0)+sum(match['neural_actions'].get(k,0) for k in ('task','report','repair','emergency','silent','share_evidence','accuse','defend','question','strategic_claim','skip','vote')):
            raise RuntimeError('Individual action accounting mismatch')
    for owner,components in rollout['reward_components'].items():
        index=int(owner.split('/')[0]);match=rollout['matches'][index]
        if components.get('exploration',0)>.3+1e-7 or components.get('new_player',0)>.7+1e-7:
            raise RuntimeError('Repeated novelty reward exploit')
        task_count=match.get('assigned_tasks_per_actor',{}).get(owner.split('/',1)[1],4)
        task_reward=match.get('task_reward_coefficient',C['task'])
        if components.get('task',0)+components.get('ghost_task',0)>task_count*max(task_reward,C['ghost_task'])+1e-7:
            raise RuntimeError('Duplicated task reward exploit')
        if not any(isclose(components.get('death',0),v,abs_tol=1e-7) for v in (0,C['death'])) or components.get('report',0)>C['report']*match['crew_reports']+1e-7:
            raise RuntimeError('Duplicated death/report accounting')
        if C['ghost_task']==0 and abs(components.get('ghost_task',0))>1e-7:raise RuntimeError('Ghost task direct reward must be zero')
        expected=match.get('terminal_reward_expected',10 if match['winner']=='crewmate' else -10 if match['winner']=='impostor' else 0)
        if not isclose(components.get('win',0)+components.get('loss',0),expected,abs_tol=1e-7):
            raise RuntimeError('Terminal owner reward attribution mismatch')
        if match.get('reward_regime')=='investigation-a2-v1':
            pid=owner.split('/',1)[1]
            expected=sum(b['reward'] for b in match.get('accepted_ballots',[]) if b['voter_id']==pid)
            if not isclose(components.get('vote_accuracy',0),expected,abs_tol=1e-7):raise RuntimeError('Accepted-ballot credit attribution mismatch')


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
