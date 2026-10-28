"""Frozen-policy development validation. Truth labels never enter policy packets."""
from collections import Counter
import math
import numpy as np
from crewmate_v2.config import MatchMixture
from social_deduction.actor import Role
from social_deduction.phase3_api import IntentKind
from tools.reporting_diversity_v2 import ExposureTracker,reporter_weights,report_context
from .config import match_seed
from .env import LearningEnv
from .policy import tensors


def forecast(packet,options,belief,observation,distribution):
    """Actor-only forecasts, computed before hidden scoring labels are inspected."""
    choices=[(i,c.target_player) for i,c in enumerate(options) if c.kind=='vote' and c.target_player!=observation.own.player_id]
    if not choices:return None
    ids=[p for i,p in choices]
    values=np.asarray([float(distribution.probs[0,i]) for i,p in choices],float)
    belief_values=np.asarray([belief.by_player[p] for p in ids],float)
    reporter,_=report_context(observation)
    conditional=distribution.logits[0,[i for i,p in choices]].double().softmax(0).tolist()
    return dict(candidates=ids,reporter=reporter,policy_vote_ranking=conditional,candidate_vote_mass=float(values.sum()),
        vote_probability=sum(float(distribution.probs[0,i]) for i,c in enumerate(options) if c.kind=='vote'),
        self_vote_probability=sum(float(distribution.probs[0,i]) for i,c in enumerate(options) if c.kind=='vote' and c.target_player==observation.own.player_id),
        skip_probability=sum(float(distribution.probs[0,i]) for i,c in enumerate(options) if c.kind=='skip'),
        frozen_belief=(belief_values/belief_values.sum()).tolist(),
        reporter_heuristics=reporter_weights(ids,reporter))


def probability_metrics(predictions):
    if not predictions:return dict(n=0,top1=None,nll=None,brier=None,ece=None,bins=[])
    bins=[dict(n=0,confidence=0.,correct=0.) for _ in range(10)];correct=nll=brier=0.;confident_wrong=0
    for probabilities,target in predictions:
        p=np.asarray(probabilities,float);y=np.zeros(len(p));y[target]=1.
        best=np.flatnonzero(np.isclose(p,p.max(),atol=1e-12,rtol=0.))
        hit=float(target in best)/len(best);correct+=hit;nll-=math.log(max(p[target],1e-12));brier+=float(((p-y)**2).sum())
        confident_wrong+=float(p.max()>=.9 and hit==0.)
        slot=min(9,int(p.max()*10));bins[slot]['n']+=1;bins[slot]['confidence']+=float(p.max());bins[slot]['correct']+=hit
    ece=0.
    for row in bins:
        if row['n']:
            row['confidence']/=row['n'];row['correct']/=row['n']
            ece+=row['n']/len(predictions)*abs(row['confidence']-row['correct'])
    return dict(n=len(predictions),top1=correct/len(predictions),nll=nll/len(predictions),
        brier=brier/len(predictions),ece=ece,bins=bins,high_confidence_wrong=int(confident_wrong))


def evaluate_game(policy,*,seed,mode,rotation=0,opponents='varied',deadline=None,lobby=None):
    import torch
    env=LearningEnv(rewards=None,opponents=opponents,rotation=rotation,coalesce=True,
        mixture=MatchMixture(mode),deadline=deadline,lobby=lobby)
    packet,_=env.reset(seed=seed);tracker=ExposureTracker();counts=Counter();seen=set();forecasts=[]
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
        tracker.observe(obs,choice.intent)
        counts['actionable_decisions']+=1;counts['legal_report_windows']+=any(a.kind is IntentKind.REPORT for a in obs.actions)
        counts['report_submissions']+=choice.kind=='report'
        packet,_,done,_,info=env.step(action)
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
    return dict(seed=seed,mode=mode,rotation=rotation,opponents=opponents,reason=game.result.reason,
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


def validation_contexts():
    for i in range(4):
        for rotation in (0,1,2):yield dict(seed=match_seed('validation',i),mode=1.,rotation=rotation,opponents='varied')
        yield dict(seed=match_seed('validation',i),mode=1.,rotation=0,opponents='original')
        yield dict(seed=match_seed('validation',i),mode=0.,rotation=0,opponents='varied')
