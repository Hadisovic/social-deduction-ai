"""Save exact registration/readiness artifacts. Never executes a study game or starts training."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from crewmate_v2.fourcrew.registration import plan,runtime,EXPERIMENT
from crewmate_v2.fourcrew.policy import FourCrewPolicy
from crewmate_v2.fourcrew.experiment import write
from crewmate_v2.learning.provenance import digest
from crewmate_v2.learning.train import state_hash


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT).decode().strip()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spectator-source',type=Path,required=True);a=p.parse_args()
    spectator=a.spectator_source.resolve();out=ROOT/'docs/crewmate-v2-fourcrew/registration'
    bdir=ROOT/'local-runs/representative-benchmark';b=read(bdir/'benchmark.json');br=read(bdir/'registration.json')
    assert b['status']=='completed_representative_engineering_benchmark' and b['zero_unchanged']
    assert not b['study_seed_partitions_opened'] and not b['full_training_started']
    benchmark_commit=git('rev-parse','0405076')
    for name,h in br['runtime']['policy']['fourcrew'].items():
        code=subprocess.check_output(['git','show',benchmark_commit+':crewmate_v2/fourcrew/'+name],cwd=ROOT)
        assert hashlib.sha256(code.replace(b'\r\n',b'\n')).hexdigest()==h,name
    current=runtime(spectator)
    assert current['policy']['base']==br['runtime']['policy']['base'] and current['spectator']==br['runtime']['spectator']
    differences=[name for name,h in br['runtime']['policy']['fourcrew'].items() if h!=current['policy']['fourcrew'][name]]
    assert set(differences)<= {'config.py','registration.py','experiment.py','analysis.py'}
    old_cfg=git('show',benchmark_commit+':crewmate_v2/fourcrew/config.py');new_cfg=(ROOT/'crewmate_v2/fourcrew/config.py').read_text()
    reward_class=lambda text:ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='Rewards'),include_attributes=False)
    assert reward_class(old_cfg)==reward_class(new_cfg),'Reward coefficients changed'
    torch.set_num_threads(1);models=[]
    for _ in range(3):torch.manual_seed(17);models.append(FourCrewPolicy())
    initial=state_hash(models[0]);assert all(state_hash(x)==initial for x in models) and initial==plan()['initial_policy_sha256']
    models[2].eval().requires_grad_(False)
    reference=ROOT/'local-runs/readiness-reference/zero-initialization17.pt'
    models[2].save(reference,dict(registration_only=True,initialization=17,updates=0,initial_sha256=initial))
    loaded,_=FourCrewPolicy.load(reference);assert state_hash(loaded)==initial
    write(out/'plan.json',plan())
    write(out/'approval-template.json',dict(approved=False,experiment=EXPERIMENT,plan_sha256=digest(plan()),
        runtime_sha256=digest(current),instruction='Do not change approved to true without explicit final human authorization.'))
    write(out/'runtime-lock.json',current)
    watch=ROOT/'local-runs/live-watch'/EXPERIMENT; (watch/'replays').mkdir(parents=True,exist_ok=False)
    write(watch/'status.json',dict(status='registered_awaiting_final_owner_authorization',condition=EXPERIMENT,
        stage='not started',decisions={'original':0,'shaped_v2':0},decision_target=2048,validation_games=0,validation_target=48,elapsed_seconds=0))
    initial_replay=bdir/'replays/benchmark-ordinary/ordinary-2760020.jsonl'
    launch=f"Set-Location '{spectator.as_posix()}'\n& 'C:/Files/RL/.venv/Scripts/python.exe' './run_training_spectator.py' --replay '{initial_replay.as_posix()}' --watch '{(watch/'replays').as_posix()}' --status-file '{(watch/'status.json').as_posix()}' --mode auto_next --speed 4\n"
    with (watch/'launch-spectator.ps1').open('x',encoding='utf-8') as stream:stream.write(launch)
    assert not (ROOT/'local-runs'/EXPERIMENT).exists(),'Actual study output unexpectedly exists'
    times={r['context']:r['wall_seconds'] for r in b['matches']};factor=b['batch_inference_wall_seconds']/(sum(times.values())/4)
    evaluation_work=24*times['ordinary']+12*times['body']+6*times['meeting_crew']+6*times['meeting_impostor']
    estimates={}
    for matches in (42,64):
        work=2*matches*max(times['ordinary'],times['body'])+evaluation_work
        estimates[str(matches)]=(work/4*max(1.,factor)*1.5+6144/51*b['ppo_update_seconds'])/60
    def tests(path):
        suites=list(ET.parse(path).getroot().iter('testsuite'))
        result={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
        assert result['failures']==result['errors']==result['skipped']==0
        return result
    readiness=dict(status='registered_ready_pending_final_owner_authorization',plan_sha256=digest(plan()),runtime_sha256=digest(current),
        initial_policy_sha256=initial,current_source_reference_sha256=sha(reference),zero_reference_frozen=True,
        benchmark=dict(source_commit=benchmark_commit,wall_seconds=b['seconds'],parallel_collection_seconds=b['batch_inference_wall_seconds'],
            ppo_update_seconds=b['ppo_update_seconds'],matches=[{k:r[k] for k in ('context','simulated_seconds','decisions','wall_seconds','reason')} for r in b['matches']],
            neural_reports=sum(sum(v['reporting']['accepted'] for v in r['actors'].values()) for r in b['matches']),
            zero_reference_unchanged=True,resources=b['resource_measurement']),
        cost_model=dict(measured_batch_factor=factor,safety_multiplier=1.5,estimated_total_minutes_by_matched_training_games_per_arm=estimates,
            hard_stop_seconds=14400,evaluation_reserve_seconds=3600,can_plan_within_ceiling=max(estimates.values())<240,
            limitations=['One mixed engineering batch; no sampled ordinary profile or 900s timeout profile.',
                        'Uses larger measured body cost as ordinary training proxy; future policy durations can differ.']),
        benchmark_compatibility=dict(changed_bookkeeping_only=differences,simulation_prefix_inference_rewards_forecast_sources_identical=True),
        tests=dict(v2_regression=tests(ROOT/'local-runs/readiness-regression.xml'),
                   final_targeted_overlapping=tests(ROOT/'local-runs/final-registration-tests.xml'),
                   spectator_existing=tests(spectator/'local-runs/registration-readiness-tests.xml'),
                   spectator_progress_overlapping=tests(spectator/'local-runs/registration-label-tests.xml')),
        study_training_started=False,study_seed_partitions_opened=False,final_test_partitions_opened=False,
        approvals=dict(full_training=False,template_approved=False),phase6_1_acceptance='pending')
    write(out/'readiness.json',readiness)
    private={p.relative_to(ROOT).as_posix():sha(p) for p in bdir.rglob('*') if p.is_file()}
    private[reference.relative_to(ROOT).as_posix()]=sha(reference)
    write(ROOT/'local-runs/readiness-private-file-hashes.json',private)
    print(json.dumps(dict(plan_sha256=readiness['plan_sha256'],runtime_sha256=readiness['runtime_sha256'],cost_minutes=estimates,
                         private_artifacts_pinned=len(private),study_started=False),indent=2))


if __name__=='__main__':main()
