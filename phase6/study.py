"""Execute exactly three seed-17 models and validation; never final evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from time import perf_counter, time
from .config import BASELINE_SHA, PARTITIONS, ROOT, digest, seeds, source_hashes
from .study_support import guard, runtime_versions, write_json
from .study_training import CONDITIONS, train_run
from .study_validation import CONTROLS, evaluate_job, play_match, select_checkpoint


def current_revision():
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    if dirty: raise ValueError('Commit and verify the runner before starting the study')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def preflight(output, sources, revision, deadline):
    from .env import Phase6Env
    from .train import load_policy
    a = train_run(Path(output)/'repro-a', 'full', sources, revision, deadline, verification=True)
    b = train_run(Path(output)/'repro-b', 'full', sources, revision, deadline, verification=True)
    keys = ('initial_state_sha256','final_state_sha256','transition_trace_sha256','used_match_seeds','simulated_seconds')
    if any(a[key] != b[key] for key in keys): raise ValueError('Training reproducibility failed')
    policy, experiment, _ = load_policy(Path(output)/'repro-a/policy.pt')
    env = Phase6Env(experiment)
    rows = [play_match(env, 'verification', policy, seeds('development',1)[0], deadline) for _ in range(2)]
    if rows[0] != rows[1]: raise ValueError('Checkpoint inference/complete-match replay failed')
    probe_jobs=[dict(output=str(Path(output)/('worker-'+method)),partition='validation',count=1,
                    method=method,sources=sources,deadline=deadline,
                    stop_file=str(Path(output)/'STOP.json')) for method in ('idle','random')]
    probes=parallel_jobs(probe_jobs)
    result = dict(status='preflight_passed', repeated_training_decisions=32,
                  training_state_and_transition_equal=True, development_rows=rows,
                  validation_worker_probe_rows=probes,
                  final_test_opened=False, sources=sources)
    write_json(Path(output)/'preflight.json', result, exclusive=True)
    return result


def parallel_jobs(jobs):
    rows = []
    # Disjoint job directories and paired seeds; processes each use one Torch thread.
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate_job, job) for job in jobs]
        try:
            for future in as_completed(futures): rows.extend(future.result())
        except BaseException:
            try: write_json(jobs[0]['stop_file'], dict(error='Validation worker failed'), exclusive=True)
            except FileExistsError: pass
            for future in futures: future.cancel()
            raise
    return rows


def run(output):
    revision = current_revision(); sources = source_hashes()
    began = perf_counter(); deadline = time()+6*3600
    guard(sources, deadline)
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    manifest = dict(stage='approved_three_conditions_seed17', baseline_sha=BASELINE_SHA,
                    revision=revision, sources=sources, initialization_seeds=[17],
                    runtime_versions=runtime_versions(),
                    conditions=CONDITIONS, decisions_per_model=8192, partitions=PARTITIONS,
                    train_seeds=seeds('train',20000), validation_seeds=seeds('validation',200),
                    selection='20 ID validation seeds: wins, own tasks, earliest checkpoint',
                    checkpoint_decisions=[2048,4096,6144,8192], validation_workers=2,
                    entropy_coefficient=.02, stage_wall_limit_seconds=6*3600,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    authorization='User approved three seed-17 models and validation only; remaining six require a report and new approval.',
                    final_test_opened=False)
    write_json(output/'manifest.json', manifest, exclusive=True)
    try:
        preflight(output/'verification', sources, revision, deadline)
        training = {}
        for condition in CONDITIONS:
            training[condition] = train_run(output/condition, condition, sources, revision,
                                             min(deadline,time()+2*3600))
        if len({r['initial_state_sha256'] for r in training.values()}) != 1:
            raise ValueError('Conditions did not use identical initialization')
        jobs = []
        for condition in CONDITIONS:
            for checkpoint in training[condition]['checkpoints']:
                jobs.append(dict(output=str(output/'screening'/f"{condition}-{checkpoint['decisions']}"),
                    partition='validation',count=20,method=f"{condition}@{checkpoint['decisions']}",
                    checkpoint=str(output/condition/checkpoint['name']),checkpoint_sha256=checkpoint['sha256'],
                    condition=condition,sources=sources,deadline=deadline))
        for job in jobs: job['stop_file'] = str(output/'STOP.json')
        screening_began = perf_counter()
        screening = parallel_jobs(jobs)
        screening_seconds = perf_counter()-screening_began
        selection = {condition:select_checkpoint(condition,screening,training[condition]['checkpoints']) for condition in CONDITIONS}
        write_json(output/'selection.json', dict(status='locked_from_validation_screening',selected=selection,
                   source_hashes=sources,screening_sha256={j['method']:digest(Path(j['output'])/'matches.jsonl') for j in jobs},
                   final_test_opened=False), exclusive=True)
        jobs = []
        for condition, checkpoint in selection.items():
            jobs.append(dict(output=str(output/'validation'/condition),partition='validation',count=200,
                method=condition,condition=condition,checkpoint=str(output/condition/checkpoint['name']),
                checkpoint_sha256=checkpoint['sha256'],sources=sources,deadline=deadline))
        for method in CONTROLS:
            jobs.append(dict(output=str(output/'validation'/method),partition='validation',count=200,
                             method=method,sources=sources,deadline=deadline))
        for job in jobs: job['stop_file'] = str(output/'STOP.json')
        validation_began = perf_counter()
        validation = parallel_jobs(jobs)
        validation_seconds = perf_counter()-validation_began
        from .study_report import build_report
        result = build_report(output,training,selection,screening,validation,perf_counter()-began,
                              screening_seconds, validation_seconds)
        guard(sources,deadline)
        write_json(output/'results.json',result,exclusive=True)
        print(json.dumps({'status':result['status'],'seconds':result['seconds'],'final_test_opened':False}),flush=True)
        return result
    except BaseException as error:
        write_json(output/'failure.json',dict(status='stage_stopped',error=repr(error),seconds=perf_counter()-began,
                   checkpoints_preserved=True,final_test_opened=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(args.output)


if __name__=='__main__': main()
