"""Matched lawful weak/direct-evidence probes; frozen inference, no selected actions."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
from crewmate_v2.fourcrew.policy import FourCrewPolicy
from crewmate_v2.learning.train import state_hash
from crewmate_v2.learning.policy import tensors
from social_deduction.actor import Phase
from social_deduction.phase3_api import IntentKind, WitnessedElimination
from .scenarios import evidence_fixture
from .run import write_json


def run(study,output):
    output.mkdir(parents=True,exist_ok=False);started=perf_counter()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    selected=json.loads((study/'study.json').read_text())['selected_checkpoints'];selected['zero']='reference-zero.pt'
    models={name:FourCrewPolicy.load(study/path)[0].requires_grad_(False) for name,path in selected.items()}
    before={name:state_hash(model) for name,model in models.items()};results=[]
    for case in ('weak_evidence','strong_evidence'):
        match,_,ids=evidence_fixture(2760074,case);actor=ids['observer'];game=match.game
        while game.phase is Phase.DISCUSSION and not any(a.kind is IntentKind.CLAIM for a in game.observe(actor).actions):match._prefix_tick()
        for phase in ('discussion','voting'):
            if phase=='voting':
                while game.phase is Phase.DISCUSSION:match._prefix_tick()
            view=match.views[actor];view.refresh();packet=view._packet()
            if game.observe(actor).context.phase.value!=phase:raise ValueError('Unexpected probe phase')
            legitimate_witness=any(type(e.payload) is WitnessedElimination for e in view.memory.events)
            for name,model in models.items():
                with torch.no_grad():distribution,value=model(tensors(packet))
                probs=distribution.probs[0];best=int(probs.argmax());choice=view.options[best]
                scores=dict(view.belief.by_player)
                results.append(dict(case=case,phase=phase,policy=name,actor=actor,
                    direct_witness=legitimate_witness,belief_impostor_scored_offline=scores[ids['impostor']],
                    argmax_kind=choice.kind,argmax_target_correct_offline=choice.target_player==ids['impostor'],
                    argmax_probability=float(probs[best]),entropy=float(distribution.entropy()[0]),
                    value=float(value[0]),candidate_count=len(view.options),
                    family_mass={kind:sum(float(probs[i]) for i,c in enumerate(view.options) if c.kind==kind)
                                 for kind in sorted({c.kind for c in view.options})},
                    legitimate_actor_packet_sha256=hashlib.sha256(b''.join(v.tobytes() for v in packet.values())).hexdigest(),
                    options=[dict(kind=c.kind,target=c.target_player,probability=float(probs[i])) for i,c in enumerate(view.options)]))
        if game.metrics['illegal_actions']:raise ValueError('Illegal probe setup')
    unchanged=all(state_hash(model)==before[name] for name,model in models.items())
    report=dict(status='engineering_inference_only',seed=2760074,seconds=perf_counter()-started,
        probes=results,checkpoints_unchanged=unchanged,training_updates=0,
        limits='Two engineered histories at one seed; no selected neural action, no outcome/causal learning claim. Shared belief predictor. Direct witness adds a legal claim candidate.')
    write_json(output/'private-probes.json',report)
    write_json(output/'aggregate.json',{**report,'probes':[{k:v for k,v in r.items() if k not in ('options','actor','legitimate_actor_packet_sha256')} for r in results]})
    print(json.dumps({k:v for k,v in report.items() if k!='probes'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.study,a.output)


if __name__=='__main__':main()
