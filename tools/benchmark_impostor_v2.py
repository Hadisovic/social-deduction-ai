"""Bounded inference-only development benchmark. Never invokes training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from crewmate_v2.config import MatchMixture
from impostor_research.env import ImpostorEnv
from impostor_research.frozen import FrozenModels
from impostor_research.policy import ImpostorPolicy
from impostor_research.protocol import development_seeds, manifest
from phase3_bots import ScriptedController, BotStyle
from social_deduction.phase3_api import Intent, IntentKind


class NonReportingHunter(ScriptedController):
    """Engineering control to expose crew discovery opportunities; not a learner."""
    def _report(self, obs): return None


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    models=FrozenModels(); network=ImpostorPolicy(seed=71).eval()
    # Small engineering controls, not validation or architecture selection.
    jobs=[dict(seed=s,mode=mode,control='hunter',fast=True,coalesce=True)
          for mode in ('ordinary','deduction') for s in development_seeds(2)]
    jobs += [dict(seed=development_seeds(1)[0],mode='ordinary',control=control,fast=True,coalesce=True)
             for control in ('idle','untrained')]
    jobs += [dict(seed=s,mode='ordinary',control='hunter_no_report',fast=True,coalesce=True)
             for s in development_seeds(2)]
    jobs += [dict(jobs[0],fast=False,coalesce=False),dict(jobs[0])]
    records=[];began=perf_counter()
    for i,job in enumerate(jobs):
        env=ImpostorEnv(models=models,mixture=MatchMixture(float(job['mode']=='deduction')),
                       fast=job['fast'],coalesce=job['coalesce'])
        if i==0:(output/'manifest.json').write_text(json.dumps(manifest(env),indent=2)+'\n',encoding='utf-8')
        start=perf_counter();packet,_=env.reset(seed=job['seed']);counts=Counter();calls=0
        script=(NonReportingHunter if job['control']=='hunter_no_report' else ScriptedController)(91,BotStyle(impostor='hunter'))
        generator=torch.Generator().manual_seed(71)  # independent of crew/model/environment streams
        while not env.finished:
            if job['control']=='untrained':
                action,_,_=network.act(packet,deterministic=False,generator=generator)
                counts[env.options[action].kind]+=1
                packet,_,_,_,_=env.step(action)
            else:
                action=script.decide(env.envelope.actor) if job['control'] in ('hunter','hunter_no_report') else Intent(IntentKind.WAIT)
                counts[action.kind.value]+=1
                packet,_,_,_,_=env.step_scripted(action)
            calls+=1
        summary=env.summary();seconds=perf_counter()-start
        record=dict(**job,**{k:v for k,v in summary.items() if k!='seed'},seconds=seconds,
                    external_decisions=calls,decisions_per_second=calls/seconds,
                    physics_ticks_per_second=env.match.game.metrics['steps']/seconds,actions=dict(counts))
        records.append(record)
        (output/'progress.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(job=i+1,total=len(jobs),**record)),flush=True)
    keys=('event_hash','action_hash','actor_hash','reason','kills','simulated_seconds')
    if not all(records[0][k]==records[-1][k]==records[-2][k] for k in keys):
        raise AssertionError('Reference/optimized/repeat engineering traces differ')
    models.assert_unchanged()
    result=dict(status='engineering_only_no_training',jobs=records,seconds=perf_counter()-began,
                exact_reference_and_repeat=True,frozen_weights_unchanged=True,
                network_parameters=sum(p.numel() for p in network.parameters()),
                reference_to_optimized_ratio=records[-2]['seconds']/records[-1]['seconds'])
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
