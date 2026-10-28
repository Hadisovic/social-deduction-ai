"""Approved six-model continuation; immutable pilot reuse and no final tests."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter, time
from .config import ROOT, PARTITIONS, Experiment, digest, source_hashes
from .study import current_revision, parallel_jobs, preflight
from .study_support import guard, runtime_versions, write_json
from .study_training import CONDITIONS, train_run
from .study_validation import CONTROLS, select_checkpoint

SEEDS = (29, 43)
CEILING_SECONDS = 6.42 * 3600
PILOT = ROOT/'docs/benchmarks/phase6/seed17-stage'
EXTENSION_FILES = {'phase6/study.py', 'phase6/study_training.py',
                   'phase6/study_validation.py', 'phase6/study_continuation.py',
                   'phase6/study_comparison.py'}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verify_pilot(sources):
    manifest = json.loads((PILOT/'manifest.json').read_text())
    selection = json.loads((PILOT/'selection.json').read_text())
    if runtime_versions() != manifest['runtime_versions']:
        raise ValueError('Pilot dependency/runtime versions differ; review required')
    changed = {p for p in sources.keys() | manifest['sources'].keys()
               if sources.get(p) != manifest['sources'].get(p)}
    if not changed <= EXTENSION_FILES:
        raise ValueError(f'Frozen scientific sources changed: {changed - EXTENSION_FILES}')
    if selection['source_hashes'] != manifest['sources']:
        raise ValueError('Pilot selection/source mismatch')
    copies = json.loads((PILOT/'checkpoint-copies.json').read_text())
    evidence = {}
    for entries in copies['models'].values():
        for entry in entries:
            if digest(ROOT/entry['path']) != entry['sha256']:
                raise ValueError('Preserved pilot checkpoint changed')
            evidence[entry['path']] = entry['sha256']
    for folder, count in (('screening', 20), ('validation', 200)):
        expected = ({f'{c}-{step}' for c in CONDITIONS for step in (2048,4096,6144,8192)}
                    if folder == 'screening' else set(CONDITIONS+CONTROLS))
        if {p.name for p in (PILOT/folder).iterdir() if p.is_dir()} != expected:
            raise ValueError('Pilot evidence job set differs')
        for p in (PILOT/folder).glob('*/matches.jsonl'):
            complete = json.loads((p.parent/'complete.json').read_text())
            data=rows(p)
            if (digest(p) != complete['matches_sha256'] or len(data) != count
                    or [r['seed'] for r in data] != list(range(1220000,1220000+count))
                    or any(r['family']!='ID' or r['illegal_actions'] or r['navigation_failures'] for r in data)):
                raise ValueError('Pilot evidence incomplete or modified')
            evidence[p.relative_to(ROOT).as_posix()] = digest(p)
    # Bind all pilot records, configurations, weights and selection locks to this stage.
    for p in PILOT.rglob('*'):
        if p.is_file(): evidence[p.relative_to(ROOT).as_posix()] = digest(p)
    return dict(original_manifest=manifest, changed_sources=sorted(changed),
                preserved_sha256=evidence, final_test_opened=False)


def budget_projection(output, began, training, remaining, pilot, validation_seconds):
    observed = sum(t['seconds'] for t in training.values())
    expected_done = sum(pilot['details'][key.rsplit('-', 1)[0]]['training']['seconds']
                        for key in training)
    scale = observed/expected_done if expected_done else 1.
    forecast = (perf_counter()-began + validation_seconds +
                scale*sum(pilot['details'][condition]['training']['seconds']
                          for seed, condition in remaining))
    write_json(Path(output)/'budget.json', dict(seconds=perf_counter()-began,
        projected_total_seconds=forecast, ceiling_seconds=CEILING_SECONDS))
    if forecast > CEILING_SECONDS:
        raise TimeoutError('Projected stage exceeds 6.42h; stop for approval')


def run(output):
    revision = current_revision(); sources = source_hashes()
    preserved = verify_pilot(sources)
    pilot = json.loads((PILOT/'results.json').read_text())
    began = perf_counter(); deadline = time()+CEILING_SECONDS
    guard(sources, deadline)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    manifest = dict(stage='approved_remaining_six', revision=revision,
        research_parent='8187d51b6bbeb506fd2a59f25504adb368ab5006', sources=sources,
        initialization_seeds=list(SEEDS), conditions=list(CONDITIONS), decisions_per_model=8192,
        partitions=PARTITIONS, experiment={c:Experiment(memory=c).to_dict() for c in CONDITIONS},
        checkpoint_decisions=[2048,4096,6144,8192], entropy_coefficient=.02, gamma=.99,
        runtime_versions=runtime_versions(), training_workers=1, validation_workers=2,
        ceiling_seconds=CEILING_SECONDS, started_utc=datetime.now(timezone.utc).isoformat(),
        pilot_provenance=preserved, final_test_opened=False,
        authorization='User approved remaining six models, unchanged protocol; 6.42h ceiling; stop before final evaluation.')
    write_json(output/'manifest.json', manifest, exclusive=True)
    plan = [(seed,c) for seed in SEEDS for c in CONDITIONS]
    validation_estimate = pilot['updated_remaining_six']['validation_seconds_estimate']
    training = {}
    try:
        for seed in SEEDS:
            preflight(output/'verification'/str(seed), sources, revision, deadline,
                      initialization_seed=seed)
        for i,(seed,condition) in enumerate(plan):
            budget_projection(output,began,training,plan[i:],pilot,validation_estimate)
            key = f'{condition}-{seed}'
            training[key] = train_run(output/key,condition,sources,revision,
                min(deadline,time()+2*3600),initialization_seed=seed)
        for seed in SEEDS:
            if len({training[f'{c}-{seed}']['initial_state_sha256'] for c in CONDITIONS}) != 1:
                raise ValueError('Conditions do not share matched initialization')
        if training['full-29']['initial_state_sha256'] == training['full-43']['initial_state_sha256']:
            raise ValueError('Independent initialization seeds did not differ')
        budget_projection(output,began,training,[],pilot,validation_estimate)
        selections = {}; screening_seconds = 0.
        for seed in SEEDS:
            jobs=[]
            for condition in CONDITIONS:
                for cp in training[f'{condition}-{seed}']['checkpoints']:
                    jobs.append(dict(output=str(output/'screening'/f'{condition}-{seed}-{cp["decisions"]}'),
                        partition='validation',count=20,method=f'{condition}@{cp["decisions"]}',
                        checkpoint=str(output/f'{condition}-{seed}'/cp['name']),checkpoint_sha256=cp['sha256'],
                        condition=condition,initialization_seed=seed,sources=sources,deadline=deadline,
                        stop_file=str(output/'STOP.json')))
            start=perf_counter();screening=parallel_jobs(jobs);screening_seconds+=perf_counter()-start
            for c in CONDITIONS:
                selections[f'{c}-{seed}']=select_checkpoint(c,screening,training[f'{c}-{seed}']['checkpoints'])
        write_json(output/'selection.json',dict(status='locked_from_validation_screening',
            selected=selections,sources=sources,screening_sha256={p.parent.name:digest(p)
                for p in (output/'screening').glob('*/matches.jsonl')},final_test_opened=False),exclusive=True)
        budget_projection(output,began,training,[],pilot,1200*screening_seconds/480)
        jobs=[]
        for seed,c in plan:
            key=f'{c}-{seed}';cp=selections[key]
            jobs.append(dict(output=str(output/'validation'/key),partition='validation',count=200,
                method=c,condition=c,initialization_seed=seed,checkpoint=str(output/key/cp['name']),
                checkpoint_sha256=cp['sha256'],sources=sources,deadline=deadline,stop_file=str(output/'STOP.json')))
        start=perf_counter();parallel_jobs(jobs);validation_seconds=perf_counter()-start
        for key,cp in selections.items():
            a=rows(output/'screening'/f'{key}-{cp["decisions"]}'/'matches.jsonl')
            b=rows(output/'validation'/key/'matches.jsonl')[:20]
            if [dict(r,method='same') for r in a] != [dict(r,method='same') for r in b]:
                raise ValueError('Selected checkpoint replay mismatch')
        if verify_pilot(sources) != preserved: raise ValueError('Preserved pilot evidence changed')
        guard(sources,deadline)
        from .study_comparison import build_comparison
        result=build_comparison(output,training,selections,pilot,perf_counter()-began,
                                screening_seconds,validation_seconds)
        guard(sources,deadline)
        write_json(output/'results.json',result,exclusive=True)
        print(json.dumps(dict(status=result['status'],seconds=result['seconds'],final_test_opened=False)),flush=True)
        return result
    except BaseException as error:
        write_json(output/'failure.json',dict(status='stopped_preserving_checkpoints',error=repr(error),
            seconds=perf_counter()-began,ceiling_seconds=CEILING_SECONDS,final_test_opened=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
