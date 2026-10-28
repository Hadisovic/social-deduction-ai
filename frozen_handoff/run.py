"""Run four frozen historical crewmates; no optimizer or model updates."""
import argparse
import importlib
import json
from pathlib import Path
from time import perf_counter
import torch
from crewmate_v2.learning.train import state_hash
from spectator.recorder import MatchRecorder
from .models import load_model,MODELS,ROOT,sha256
from .match import FrozenCrewMatch


def run(*,model='B',seed=2760000,greedy=False,replay=None,controller_factory=None,
        accounting='historical',opponent='baseline',wall_seconds=600):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    policy,metadata=load_model(model);before=state_hash(policy);began=perf_counter()
    match=FrozenCrewMatch(seed,model=model,controller_factory=controller_factory,accounting=accounting,
                         opponent=opponent,deadline=began+wall_seconds)
    recorder=None
    if replay:
        recorder=MatchRecorder(match.match,replay,checkpoint_path=MODELS[model][0],
            training_session='frozen-five-player-handoff',model_architecture='Historical FourCrewPolicy, 51778 parameters',
            controller_types={pid:'frozen' if pid in match.crew else 'external-impostor' if controller_factory else 'scripted' for pid in match.game.players},
            metadata=dict(training_condition='Frozen Model '+model,context='ordinary',policy_version=-1,
                handoff_options=match.handoff_options,lobby=match.game.lobby.to_dict(),
                reward_profile=next(iter(match.ledgers.values())).config.to_dict(),
                frozen_belief='shared Phase 4 predictor, independent per-actor histories'))
    try:
        _,summary=match.collect(policy,initialization=17,deterministic=greedy,recorder=recorder)
    finally:
        if recorder and not recorder.recorder._closed:recorder.recorder._stream.close();recorder.recorder._closed=True
    if state_hash(policy)!=before:raise AssertionError('Frozen policy mutated')
    # Semantic event hashes exclude wall-clock performance counters, not outcomes.
    import hashlib
    events=json.dumps(match.game.event_log,sort_keys=True,separators=(',',':'))
    summary.update(model=model,sampling='greedy' if greedy else 'sampled',seconds=perf_counter()-began,
        winner=match.game.result.winner.value if match.game.result.winner is not None else None,
        checkpoint_sha256=sha256(ROOT/MODELS[model][0]),weights_unchanged=True,options=match.handoff_options,
        event_sha256=hashlib.sha256(events.encode()).hexdigest(),
        independent_actor_memories=len({id(v.memory) for v in match.views.values()})==4,
        independent_public_memories=len({id(v.public_memory) for v in match.views.values()})==4,
        shared_frozen_belief=len({id(v.belief_model) for v in match.views.values()})==1,
        impostor_controller_calls=match.match.controllers[match.impostor].calls)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('A','B'),default='B');p.add_argument('--seed',type=int,default=2760000)
    p.add_argument('--greedy',action='store_true');p.add_argument('--replay',type=Path);p.add_argument('--summary',type=Path)
    p.add_argument('--accounting',choices=('historical','eligible-v1'),default='historical')
    p.add_argument('--opponent',choices=('baseline','v1','v2'),default='baseline')
    p.add_argument('--controller',help='Importable module:factory; factory(seed=..., player_id=...) returns decide(observation)')
    a=p.parse_args();factory=None
    if a.controller:
        module,attribute=a.controller.rsplit(':',1);factory=getattr(importlib.import_module(module),attribute)
    summary=run(model=a.model,seed=a.seed,greedy=a.greedy,replay=a.replay,controller_factory=factory,
                accounting=a.accounting,opponent=a.opponent)
    if a.summary:a.summary.parent.mkdir(parents=True,exist_ok=True);a.summary.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
