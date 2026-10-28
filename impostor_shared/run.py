"""Complete bounded integration matches and record real replay telemetry. No PPO."""
import argparse,hashlib,json
from pathlib import Path
from time import perf_counter
import torch
from frozen_handoff.match import FrozenCrewMatch
from frozen_handoff.models import load_model,MODELS,ROOT,sha256
from crewmate_v2.learning.train import state_hash
from spectator.recorder import MatchRecorder
from .controller import Controller
from .coordinator import ContextGate,ImpostorLedger
from .weights import load_transfer,verify_parity,tensor_hash

PUBLICATION='c3320583f7d562bc69ea527580a91ddc5db54f18'

def run(*,output,seed=2760002,crew='B',stochastic=False,recording=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    began=perf_counter();policy,metadata=load_model(crew);impostor,receipt=load_transfer()
    adapter_sources={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    parity=verify_parity(impostor,receipt)
    frozen_before=state_hash(policy);imp_before=tensor_hash(impostor)
    actors=[]
    def factory(*,seed,player_id):
        actor=Controller(impostor,seed=seed,player_id=player_id,stochastic=stochastic);actors.append(actor);return actor
    match=FrozenCrewMatch(seed,model=crew,controller_factory=factory,accounting='eligible-v1',deadline=began+600)
    actor=actors[0];belief_before=state_hash(next(iter(match.views.values())).belief_model.network)
    recorder=None
    if recording:
        recorder=MatchRecorder(match.match,output/'replay.jsonl',source_version=PUBLICATION,
            training_session='Shared integration: frozen '+crew+' x4 vs V4.1 transfer83/8192; zero updates',
            model_architecture='Crew FourCrewPolicy51778; impostor FamilyPolicy75075 (explicit transfer)',
            controller_types={pid:'frozen-neural-'+crew if pid in match.crew else 'learned-impostor-transfer' for pid in match.game.players},
            metadata=dict(training_condition='Frozen '+crew+' x4 versus transferred V4.1',context='ordinary-engineering',
                agent_models={pid:MODELS[crew][0] if pid in match.crew else 'impostor_shared/assets-v1/policy-transfer.pt' for pid in match.game.players},
                crew_checkpoint_sha256=MODELS[crew][1],impostor_original_sha256=receipt['source_checkpoint_sha256'],
                impostor_transfer_sha256=receipt['export_sha256'],accounting='eligible-v1',optimizer_updates=0,
                impostor_objective='V4.1 coefficients; shared-adapter-v1 timing/projection',social_guards=False))
    ledger=ImpostorLedger(match.game,actor,recorder)
    match.match.controllers[match.impostor]=ContextGate(match.game,actor,ledger)
    match.game.step=ledger.step  # Instance-only trusted audit hook; published sources unchanged.
    try:
        _,summary=match.collect(policy,initialization=17,recorder=recorder)
        credit=ledger.check()
    finally:
        if recorder and not recorder.recorder._closed:recorder.recorder._stream.close();recorder.recorder._closed=True
    after=state_hash(policy);belief_after=state_hash(next(iter(match.views.values())).belief_model.network)
    if frozen_before!=after or belief_before!=belief_after or tensor_hash(impostor)!=imp_before:raise AssertionError('Inference changed weights')
    if any(p.requires_grad or p.grad is not None for p in policy.parameters()):raise AssertionError('Frozen crew gradient leak')
    if any(p.requires_grad or p.grad is not None for p in next(iter(match.views.values())).belief_model.network.parameters()):raise AssertionError('Belief gradient leak')
    events=json.dumps(match.game.event_log,sort_keys=True,separators=(',',':'))
    summary.update(status='complete_engineering_zero_updates',publication=PUBLICATION,model=crew,adapter_sources=adapter_sources,
        crew_checkpoint_sha256=sha256(ROOT/MODELS[crew][0]),impostor_original_sha256=receipt['source_checkpoint_sha256'],
        transfer_sha256=receipt['export_sha256'],impostor_tensor_hash=imp_before,parity=parity,credit=credit,
        winner=match.game.result.winner.value if match.game.result.winner else None,
        seconds=perf_counter()-began,event_sha256=hashlib.sha256(events.encode()).hexdigest(),
        impostor_actions=dict(actor.actions),impostor_neural_decisions=actor.decisions,held_callbacks=actor.held_callbacks,
        fake_repeat_continuations=actor.fake_repeats,weights_unchanged=True,social_guards=False,
        independent_actor_memories=len({id(v.memory) for v in match.views.values()})==4,
        independent_public_memories=len({id(v.public_memory) for v in match.views.values()})==4,
        shared_frozen_belief=len({id(v.belief_model) for v in match.views.values()})==1,
        replay='replay.jsonl' if recording else None,full_training=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed',type=int,default=2760002);p.add_argument('--crew',choices=('A','B'),default='B')
    p.add_argument('--stochastic',action='store_true');p.add_argument('--no-replay',action='store_true')
    a=p.parse_args();result=run(output=a.output,seed=a.seed,crew=a.crew,stochastic=a.stochastic,recording=not a.no_replay)
    print(json.dumps({k:result[k] for k in ('status','seed','model','winner','reason','simulated_seconds','seconds','impostor_actions','credit','event_sha256','illegal_actions')},indent=2))

if __name__=='__main__':main()
