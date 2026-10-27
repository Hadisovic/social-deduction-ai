"""Same frozen validation decisions/scenarios, plus offline action/evidence auditing."""
from collections import Counter
import numpy as np
from crewmate_v2.config import MatchMixture
from crewmate_v2.learning.config import RewardConfig
from crewmate_v2.learning.env import LearningEnv
from crewmate_v2.learning.policy import tensors
from crewmate_v2.learning.evaluate import forecast,probability_metrics
from social_deduction.actor import Role
from social_deduction.phase3_api import IntentKind,WitnessedElimination
from tools.reporting_diversity_v2 import ExposureTracker

def evaluate_game(policy,*,seed,mode,rotation=0,opponents='varied',deadline=None,lobby=None):
    import torch
    env=LearningEnv(rewards=None,opponents=opponents,rotation=rotation,coalesce=True,
        mixture=MatchMixture(mode),deadline=deadline,lobby=lobby)
    packet,_=env.reset(seed=seed);tracker=ExposureTracker();counts=Counter();seen=set();forecasts=[]
    action_counts=Counter();vote_evidence={key:Counter() for key in ("with_direct_witness","without_direct_witness")}
    while True:
        obs=env.observation
        # One pre-vote forecast per focal meeting; model consumes no reward/info.
        with torch.no_grad():distribution,_=policy(tensors(packet))
        prediction=forecast(packet,env.options,env.belief,obs,distribution)
        if prediction is not None and obs.context.meeting_id not in seen:
            probe={k:v.copy() for k,v in packet.items()};probe['players'][:,11]=0.
            with torch.no_grad():probe_distribution,_=policy(tensors(probe))
            prediction['report_counter_channel_tv']=float((distribution.probs-probe_distribution.probs).abs().sum()/2)
            seen.add(obs.context.meeting_id);forecasts.append(prediction)
        action=int(distribution.probs.argmax(-1).item());choice=env.options[action]
        direct=any(type(e.payload) is WitnessedElimination for e in env.memory.events)
        evidence_key='with_direct_witness' if direct else 'without_direct_witness'
        action_counts[choice.kind]+=1
        tracker.observe(obs,choice.intent)
        counts['actionable_decisions']+=1;counts['legal_report_windows']+=any(a.kind is IntentKind.REPORT for a in obs.actions)
        counts['report_submissions']+=choice.kind=='report'
        packet,_,done,_,info=env.step(action)
        for key in ('correct_votes','incorrect_votes','skips','automatic_skips'):
            vote_evidence[evidence_key][key]+=info['reward_counts'].get(key,0)
        counts['forced_windows']+=info['coalesced_forced_windows']
        if done:break
    game=env.match.game;belief_predictions=[];policy_predictions=[];heuristic=Counter();strata={}
    for f in forecasts:
        target=next(i for i,pid in enumerate(f['candidates']) if game.players[pid].role is Role.IMPOSTOR)
        belief_predictions.append((f['frozen_belief'],target));policy_predictions.append((f['policy_vote_ranking'],target))
        role=game.players[f['reporter']].role.value if f['reporter'] is not None else 'emergency'
        strata.setdefault(role,[]).append((f['policy_vote_ranking'],target))
        for kind,weights in f['reporter_heuristics'].items():heuristic[kind]+=weights[f['candidates'][target]]
    ledger=env.match.ledger
    result=dict(seed=seed,mode=mode,rotation=rotation,opponents=opponents,reason=game.result.reason,
        evaluation_reward='common legacy score; all accepted-action counts also recorded',
        crew_win=game.result.winner is Role.CREWMATE,simulated_seconds=game.time,behavior=dict(counts),
        reward_event_counts=dict(ledger.counts),reward_components=dict(ledger.components),
        body_visible_unique=len(tracker.pairs),report_legal_unique=sum(p['first_legal'] is not None for p in tracker.pairs.values()),
        physical_body_visible_unique=len(env.match.report_visible),physical_report_legal_unique=len(env.match.report_legal),
        scheduled_report_legal_windows=env.match.report_legal_windows,
        pre_vote_count=len(forecasts),frozen_belief=probability_metrics(belief_predictions),
        mean_vote_probability=float(np.mean([f['vote_probability'] for f in forecasts])) if forecasts else None,
        mean_skip_probability=float(np.mean([f['skip_probability'] for f in forecasts])) if forecasts else None,
        mean_self_vote_probability=float(np.mean([f['self_vote_probability'] for f in forecasts])) if forecasts else None,
        mean_report_counter_channel_tv=float(np.mean([f['report_counter_channel_tv'] for f in forecasts])) if forecasts else None,
        policy_vote_ranking_by_reporter_role={k:probability_metrics(v) for k,v in strata.items()},
        policy_vote_ranking=probability_metrics(policy_predictions),reporter_only_expected_correct=dict(heuristic),
        illegal_actions=game.metrics['illegal_actions'],navigation_failures=game.metrics['navigation_failures'])
    profile=RewardConfig()
    shaped=sum(getattr(profile,{'alive_tasks':'alive_task','ghost_tasks':'ghost_task','reports':'report','correct_votes':'correct_vote','incorrect_votes':'incorrect_vote','skips':'skip'}.get(k,k))*v
               for k,v in ledger.counts.items() if k!='automatic_skips')+ledger.components['time']
    result.update(action_counts=dict(action_counts),voting_evidence={k:dict(v) for k,v in vote_evidence.items()},
                  evidence_note='Direct witness means a retained legitimate WitnessedElimination; its absence does not exclude body/claim/report evidence.',
                  focal_survived=bool(game.players[env.focal].active),episode=info['episode'],
                  common_legacy_return=sum(ledger.components.values()),counterfactual_shaped_return=shaped,
                  repeated_correct_vote_count=max(0,ledger.counts['correct_votes']-1))
    return result
