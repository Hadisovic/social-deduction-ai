"""Paired engineering inference with/without forced cache eviction; never PPO."""
import argparse
import json
import gzip
import hashlib
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def compare_prefixes(root):
    """Compare recorded input/action/physics data at the same logical boundary."""
    root=Path(root)
    def decisions(name):
        with gzip.open(root/(name+'.decisions.jsonl.gz'),'rt',encoding='utf-8') as f:
            return [r for line in f if (r:=json.loads(line))['record']=='decision']
    a=decisions('normal');b=decisions('geometry-evictions');n=min(len(a),len(b))
    if not n:raise ValueError('No common recorded neural prefix')
    end=min(a[n-1]['tick'],b[n-1]['tick'])
    a=[r for r in a[:n] if r['tick']<=end];b=[r for r in b[:n] if r['tick']<=end]
    def replay(name,kind):
        with (root/(name+'.jsonl')).open(encoding='utf-8') as f:
            return [r for line in f if (r:=json.loads(line))['record']==kind and r.get('tick',0)<=end]
    def digest(rows):
        return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(common_decisions=len(a),last_common_time=a[-1]['time'],last_common_tick=end,
        impostor_input_action_sha256=[digest(a),digest(b)],impostor_inputs_actions_equal=a==b,
        all_recorded_neural_equal=replay('normal','neural')==replay('geometry-evictions','neural'),
        physics_equal=replay('normal','tick')==replay('geometry-evictions','tick'))


def assess(result,root):
    prefix=compare_prefixes(root);rows=result['trials'];result['common_prefix']=prefix
    evictions=rows[1]['forced_geometry_evictions']
    after=bool(evictions) and prefix['last_common_time']>=evictions[0]['tick_time']+.6-1e-8
    result['prefix_includes_post_eviction_observation']=after
    result['target_120_decisions_reached_in_both']=all(r['observations']>=120 for r in rows)
    result['assessment']='Common logical prefix after forced eviction; unequal wall-cut tails are not compared'
    result['passed']=(prefix['impostor_inputs_actions_equal'] and prefix['all_recorded_neural_equal']
        and prefix['physics_equal'] and after and result['policy_tensors_unchanged'] and result['crew_tensors_unchanged']
        and all(r['metrics']['illegal_actions']==0 and r['metrics']['navigation_failures']==0 for r in rows))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recheck-existing',action='store_true',help='Offline prefix comparison only; preserves original verification.json')
    args=p.parse_args()
    if args.recheck_existing:
        raw=args.output/'verification.json'
        result=json.loads(raw.read_text(encoding='utf-8'))
        result['original_check']=dict(passed=result['passed'],sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
        result=assess(result,args.output);result['offline_reanalysis_only']=True
        with (args.output/'verification-prefix.json').open('x',encoding='utf-8') as f:
            json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=result['passed'],common_prefix=result['common_prefix'],no_new_inference=True)))
        return 0 if result['passed'] else 1
    if not args.source_root or not args.checkpoint:p.error('Inference requires source-root and checkpoint')
    from jane.impostor.v1.bootstrap import activate,frozen_crew
    source=activate(args.source_root)
    import torch
    torch.set_num_threads(1)
    from jane.classic_v2.settings import load_profile,receipt
    from jane.impostor.v1.model import ImpostorPolicy
    from jane.impostor.v1.run import tensor_hash
    from jane.impostor.v2.checkpoint import load
    from jane.impostor.v2.run import known_passes
    from jane.impostor.v2.session import run_match
    from patrick_jane.a2_geometry import geometry,planner
    profile=load_profile(ROOT/'jane/configs/classic-v2/default_12p.json')
    model=ImpostorPolicy(731);record=load(args.checkpoint,model,profile)
    crew=frozen_crew(source);policy_hash=tensor_hash(model);crew_hash=tensor_hash(crew)
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[];began=perf_counter()
    for name,stress in (('normal',False),('geometry-evictions',True)):
        geometry.cache_clear();planner.cache_clear();evictions=[]
        def status(value):
            if stress and value['impostor_observations']>0:
                evictions.append(dict(tick_time=value['simulation_seconds'],
                    observations=value['impostor_observations'],cache_before=str(geometry.cache_info())))
                geometry.cache_clear()
        summary,_=run_match(model,crew,profile,9700032,output=args.output/(name+'.jsonl'),
            max_observations=120,max_wall_seconds=180,status=status,
            policy_update=known_passes(record),mode='engineering-cache-coherence-inference')
        rows.append(dict(name=name,trace=summary['trace'],observations=summary['impostor_observations'],
            simulation_seconds=summary['simulation_seconds'],wall_seconds=summary['wall_seconds'],
            truncated=summary['truncated'],metrics=summary['metrics'],forced_geometry_evictions=evictions))
    result=dict(schema='lizbon-geometry-verification-v1',geometry=receipt(profile),seed=9700032,
        mode='bounded engineering inference; no PPO/validation/confirmation/final-test use',trials=rows,
        equal_actor_trace=rows[0]['trace']==rows[1]['trace'],
        policy_tensors_unchanged=tensor_hash(model)==policy_hash,crew_tensors_unchanged=tensor_hash(crew)==crew_hash,
        wall_seconds=perf_counter()-began)
    result=assess(result,args.output);passed=result['passed']
    (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=passed,equal_actor_trace=result['equal_actor_trace'],wall_seconds=result['wall_seconds'],
        forced_evictions=len(rows[1]['forced_geometry_evictions']))),flush=True)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
