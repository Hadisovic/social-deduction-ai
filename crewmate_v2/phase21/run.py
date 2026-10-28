"""Bounded frozen-policy engineering comparisons. Never starts PPO training."""
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import hashlib
import json
import torch
from crewmate_v2.fourcrew.policy import FourCrewPolicy,source_lock
from crewmate_v2.fourcrew.scenarios import ScenarioMatch
from crewmate_v2.fourcrew.config import Rewards
from crewmate_v2.config import MatchMixture
from crewmate_v2.learning.train import state_hash
from crewmate_v2.learning.provenance import digest
from crewmate_v2.live.protocol import spectator_import
from crewmate_v2.phase2.scenarios import discussion_messages
from crewmate_v2.phase2.audit import OpportunityAudit,AuditedController,install_observer
from phase5_features import mechanical_intent
from social_deduction.phase3_api import Intent,IntentKind,ClaimKind
from . import VERSION
from .config import Features,modes
from .scenarios import social_fixture,witness_fixture,clone_prefix,WITNESS_CASES
from .runtime import ExperimentalMatch
from .diagnostics import DiagnosticRecorder,summarize


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def record(module,match,path,context,mode):
    return DiagnosticRecorder(module.MatchRecorder(match.match,path,training_session=VERSION,
        source_version='runtime:'+digest(source_lock())+':'+VERSION,
        controller_types={pid:'ppo' if pid in match.crew else 'scripted' for pid in match.game.players},
        model_architecture='Frozen FourCrewPolicy plus EXPLICIT experimental decision adapter',
        metadata=dict(training_condition='B frozen + '+mode,context=context,opponent=match.features.opponent,
            engineering_only=True,policy_version=23,experimental_environment=VERSION,
            features=match.features.to_dict(),reward_accounting=match.ledgers[match.crew[0]].schema,
            reward_profile=match.ledgers[match.crew[0]].config.to_dict(),
            prefix_handoff_seconds=match.start_time,
            snapshots='from handoff; prefix event history retained' if match.start_time else 'complete match',
            probabilities='Actual hybrid action probabilities; raw NN and base/adjusted belief are separate truth-only diagnostic events.')),
        match)


def close_recorder(rec):
    if rec and not rec.recorder._closed:
        rec.recorder._stream.flush();rec.recorder._stream.close();rec.recorder._closed=True


def controlled(base,features,policy,*,deadline,module=None,path=None,context='',mode='',max_roaming_seconds=None):
    match=ExperimentalMatch.adopt(clone_prefix(base),features,deadline=deadline,stop_after_meeting=True,max_roaming_seconds=max_roaming_seconds)
    rec=record(module,match,path,context,mode) if module else None
    before=perf_counter()
    try:_,summary=match.collect(policy,deterministic=True,policy_version=23,recorder=rec)
    finally:close_recorder(rec)
    row=summarize(match,summary);row['wall_seconds']=perf_counter()-before;row['mode']=mode;row['fixture']=context
    return row,match


def natural(seed,arm,frequency,policy,deadline,module,path):
    base=ScenarioMatch(seed,mixture=MatchMixture(frequency),rewards=Rewards.original(),deadline=deadline);base.prepare_prefix()
    match=ExperimentalMatch.adopt(base,Features(opponent=arm),deadline=deadline);audit=OpportunityAudit()
    controller=match.match.controllers[match.impostor];match.match.controllers[match.impostor]=AuditedController(controller,audit)
    install_observer(match,audit)
    def choices(pid,obs,memory,belief,options,probabilities,packet,selected):
        audit.decision(obs,mechanical_intent(options[selected],obs) or Intent(IntentKind.WAIT))
    rec=record(module,match,path,'ordinary' if not frequency else 'deduction_natural','unrestricted')
    started=perf_counter()
    try:_,summary=match.collect(policy,deterministic=True,policy_version=23,recorder=rec,decision_observer=choices)
    finally:close_recorder(rec)
    row=summarize(match,summary);row.update(arm=arm,deduction_frequency=frequency,wall_seconds=perf_counter()-started,
        opportunities=audit.summary(match.game),event_hash=digest(match.game.event_log),
        retained_goal_windows=getattr(controller,'retained_goals',None),chase_updates=getattr(controller,'chase_updates',None))
    # Every published valid ballot is counted, including a same-step ejected voter.
    counts={k:sum(a['reward_counts'].get(k,0) for a in summary['actors'].values()) for k in ('correct_votes','incorrect_votes','skips','automatic_skips')}
    assert counts['correct_votes']==row['counts']['correct_votes']
    assert counts['incorrect_votes']==row['counts']['incorrect_votes']
    assert counts['skips']+counts['automatic_skips']==row['counts']['published_skips']
    row['ledger_ballots_match_replay']=True
    return row,match


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);started=perf_counter();global_end=started+1800
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    module=spectator_import(args.spectator_source)
    selected=json.loads((args.study/'study.json').read_text(encoding='utf-8'))['selected_checkpoints'];selected['zero']='reference-zero.pt'
    policies={};before={}
    for name,rel in selected.items():
        path=args.study/rel;model,_=FourCrewPolicy.load(path);model.requires_grad_(False);policies[name]=model
        before[name]=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),tensors=state_hash(model))
    policy=policies['shaped_v2'];comparison=[];witness=[];opponents=[];failure=None
    protocol=dict(version=VERSION,actor_runtime=source_lock(),features={k:v.to_dict() for k,v in modes().items()},
        code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
        seed_partition='engineering 2760090-2760099',social_seeds=[2760090+i%4 for i in range(6)],
        witness_seed=2760092,opponent_seeds=list(range(2760096,2760100)),
        social_contexts=['empty','innocent','impostor','direct','conflicting','changed'],
        policy=before,no_training=True,global_wall_seconds=1800,
        section_ceilings=dict(social=600,witness=300,opponents=900),partial_segments_are_not_complete_matches=True)
    write(args.output/'protocol.json',protocol)
    try:
        section_end=min(global_end,perf_counter()+600)
        for case_index,case in enumerate(protocol['social_contexts']):
            if perf_counter()>=section_end:raise TimeoutError('Social comparison ceiling')
            base,_,ids=social_fixture(protocol['social_seeds'][case_index],case)
            if case in ('conflicting','changed'):
                messages=({base.impostor:(ClaimKind.SUSPECT_PLAYER,ids['innocent']),
                    base.crew[0]:(ClaimKind.SUSPECT_PLAYER,base.impostor),base.crew[-1]:(ClaimKind.DEFEND_PLAYER,base.impostor)}
                    if case=='conflicting' else {ids['observer']:(ClaimKind.DEFEND_PLAYER,base.impostor),ids['witness']:(ClaimKind.SAW_ELIMINATION,base.impostor)})
                discussion_messages(base,messages)
                base.prefix['handoff_tick']=base.game.tick
            origin=digest(base.game.event_log);memories={p:v.memory.to_json() for p,v in base.views.items()}
            for mode,features in modes().items():
                row,match=controlled(base,features,policy,deadline=section_end,module=module,
                    path=args.output/'replays/social'/mode/(case+'.jsonl'),context=case,mode=mode)
                row['paired_prefix_hash']=origin;row['prefix_scripted_claims']=case in ('conflicting','changed')
                write(args.output/f'social-{case}-{mode}.json',row);write(args.output/f'private-decisions-{case}-{mode}.json',match.decision_log)
                comparison.append(row)
                assert digest(base.game.event_log)==origin and all(v.memory.to_json()==memories[p] for p,v in base.views.items())
            print(json.dumps(dict(section='social',context=case,completed=len(comparison))),flush=True)
        section_end=min(global_end,perf_counter()+300)
        for case in WITNESS_CASES:
            base,ids=witness_fixture(2760092,case,prekill_policy=policy)
            for mode,features in (('unrestricted',Features()),('witness_priority',Features(witness_priority=True,social_guard=True))):
                row,match=controlled(base,features,policy,deadline=section_end,module=module,
                    path=args.output/'replays/witness'/mode/(case+'.jsonl'),context=case,mode=mode,max_roaming_seconds=25.)
                row['focal_witness']=ids['witness'];row['primary_body']=ids['victim']
                write(args.output/f'witness-{case}-{mode}.json',row);write(args.output/f'private-witness-decisions-{case}-{mode}.json',match.decision_log)
                witness.append(row)
            print(json.dumps(dict(section='witness',context=case,completed=len(witness))),flush=True)
        section_end=min(global_end,perf_counter()+900)
        for index,seed in enumerate(protocol['opponent_seeds']):
            for arm in ('baseline','v1','v2'):
                row,match=natural(seed,arm,float(index>=2),policy,section_end,module,args.output/'replays/opponents'/arm/(str(seed)+'.jsonl'))
                write(args.output/f'opponent-{seed}-{arm}.json',row);write(args.output/f'private-opponent-decisions-{seed}-{arm}.json',match.decision_log)
                opponents.append(row);print(json.dumps(dict(section='opponents',seed=seed,arm=arm,completed=len(opponents),seconds=round(row['wall_seconds'],2))),flush=True)
    except (TimeoutError,AssertionError,ValueError) as error:
        failure=type(error).__name__+': '+str(error)
    unchanged=all(state_hash(policies[n])==before[n]['tensors'] and hashlib.sha256((args.study/selected[n]).read_bytes()).hexdigest()==before[n]['sha256'] for n in policies)
    result=dict(status='complete_engineering' if len(comparison)==36 and len(witness)==14 and len(opponents)==12 and failure is None else 'partial_inconclusive',
        seconds=perf_counter()-started,social_segments=len(comparison),witness_segments=len(witness),opponent_complete_games=len(opponents),
        failure=failure,checkpoints_unchanged=unchanged,training_updates=0,final_test_opened=False,acceptance='pending')
    write(args.output/'execution.json',result);print(json.dumps(result,indent=2))
    if failure or not unchanged:raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--study',type=Path,required=True);p.add_argument('--spectator-source',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
