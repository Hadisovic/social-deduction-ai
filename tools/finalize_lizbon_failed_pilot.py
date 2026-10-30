"""Curate an interrupted pilot honestly; no inference, training or final tests."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot',type=Path);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();pilot=args.pilot;train=pilot/'train'
    registration=read(pilot/'pilot-registration.json');native=read(train/'registration.json')
    if not (train/'failure.json').exists() or (pilot/'selection-lock.json').exists():
        raise ValueError('This recovery is only for the preserved pre-selection failure')
    rows=[json.loads(s) for s in (train/'matches.jsonl').read_text().splitlines() if s.strip()]
    failed=train/'match-002.jsonl.writing';counts=Counter();last=None;header=None;decisions=Counter()
    for line in failed.read_text(encoding='utf-8').splitlines():
        try:r=json.loads(line)
        except ValueError:break
        if r['record']=='header':header=r
        elif r['record']=='tick':last=r
        elif r['record']=='event':counts[r.get('kind','')]+=1
        elif r['record']=='neural' and header['agent_roles'].get(r['player_id'])=='impostor':
            decisions[r['action_kind']]+=1
    failed_at=datetime.fromtimestamp((train/'failure.json').stat().st_mtime,timezone.utc)
    started=datetime.fromisoformat(registration['started_utc'])
    checkpoint=train/'checkpoint-001.pt'
    result=dict(model_name='Lizbon',status='FAILED_NATIVE_GEOMETRY',study_source_commit=registration['source_commit'],
        started_utc=registration['started_utc'],failed_utc=failed_at.isoformat(),
        elapsed_to_failure_seconds=(failed_at-started).total_seconds(),
        timing_method='Registration and original failure-file timestamps; exact second-match wall total was not persisted',
        approved_budget_minutes=90,automatic_restart=False,training_matches_completed=len(rows),
        training_wins=sum(r['winner']=='impostor' for r in rows),
        counters=rows[-1]['counters'],completed_training_matches=rows,
        interrupted_match=dict(seed=header['scenario_seed'],last_valid_tick=last['tick'],
            last_valid_simulation_seconds=last['sim_time'],outcome=None,counts=dict(counts),
            impostor_decisions=dict(decisions),included_in_ppo=False),
        recovery_checkpoint=dict(file='checkpoint-001.pt',sha256=sha(checkpoint),
            role='Last complete PPO boundary; preserved research checkpoint, not selected by development',
            collection_policy='The completed first game used the initial untrained policy; match2 used update8'),
        crew_checkpoint_sha256=sha(args.source_root/'artifacts/patrick-jane/task-pursuit-v1-update102/inference.pt'),
        crew_file_unchanged=sha(args.source_root/'artifacts/patrick-jane/task-pursuit-v1-update102/inference.pt')==native['crew_inference_sha256'],
        final_in_memory_crew_hash_check='not reached after assertion; loader freezes crew parameters and prior regression checks passed',
        development_matches=0,reserved_final_matches=0,selection_lock_created=False,
        learning_sufficiency='Insufficient: one initialization, one pre-update completed game, one failed post-update game, no development comparison',
        geometry_diagnostic=read(pilot/'geometry-diagnostic.json'),hardware=registration['hardware'],runtime=native['runtime'])
    write(pilot/'failure-report.json',result)
    for path in (pilot/'pilot-status.json',train/'status.json'):
        backup=path.with_name(path.stem+'.last-collecting.json')
        if not backup.exists():shutil.copy2(path,backup)
        write(path,dict(phase='failed',stage='train',failure='native dynamic-door invariant',
            failed_utc=result['failed_utc'],counters=result['counters'],restart_authorized=False,
            failure_report='failure-report.json',old_status_preserved=backup.name,
            monitor_dir=str((train/'live').resolve()),worker_count=1))
    docs=ROOT/'docs/jane-impostor-v1/pilot-20261030';docs.mkdir(parents=True,exist_ok=True)
    write(docs/'results.json',result)
    write(docs/'geometry-diagnostic.json',result['geometry_diagnostic'])
    for name in ('pilot-registration.json',):shutil.copy2(pilot/name,docs/name)
    shutil.copy2(train/'ppo.json',docs/'ppo.json')
    shutil.copy2(train/'behavior.json',docs/'behavior.json')
    for graph in (train/'graphs').glob('*.png'):
        target=docs/'graphs'/graph.name;target.parent.mkdir(exist_ok=True);shutil.copy2(graph,target)
    release=ROOT/'artifacts/lizbon-v1/pilot-20261030';release.mkdir(parents=True,exist_ok=True)
    files={}
    for name in ('initial.pt','checkpoint-001.pt'):
        target=release/name
        if target.exists() and sha(target)!=sha(train/name):raise ValueError('Refusing to overwrite different release weights')
        if not target.exists():shutil.copy2(train/name,target)
        files[name]=dict(sha256=sha(target),bytes=target.stat().st_size)
    write(release/'manifest.json',dict(model='Lizbon',status='interrupted pilot; not a validated release',
        source_commit=registration['source_commit'],engine_commit=native['source_implementation'],
        architecture=native['architecture'],settings_sha256=native['settings_sha256'],
        shared_bundle=native['shared_bundle'],runtime=native['runtime'],files=files,
        counters=result['counters'],selection='No development selection; recovery boundary only',
        final_sets_opened=False,report='docs/jane-impostor-v1/pilot-20261030/README.md'))
    original=(args.source_root/'patrick_jane/environment.py').read_text(encoding='utf-8')
    old='        assert all(self.map.contains(p.position) for p in self.players.values())'
    replacement='''        # Stored inactive/underground positions need not avoid closed surface doors.
        # Living surface actors must still satisfy the current collision geometry.
        base = getattr(self, 'base_map', self.map)
        underground = getattr(self, 'vented', {})
        assert all((base if not p.active or pid in underground else self.map).contains(p.position)
                   for pid, p in self.players.items())'''
    if original.count(old)!=1:raise ValueError('Native proposal source anchor mismatch')
    patch=''.join(difflib.unified_diff(original.splitlines(True),original.replace(old,replacement).splitlines(True),
        fromfile='a/patrick_jane/environment.py',tofile='b/patrick_jane/environment.py'))
    (docs/'PROPOSED_NATIVE_FIX.patch').write_text(patch,encoding='utf-8')
    print(json.dumps(dict(report=str(docs),release=str(release),elapsed_seconds=result['elapsed_to_failure_seconds'],
        completed=len(rows),failed_sim=last['sim_time'],partial_kills=counts['elimination'])))


if __name__=='__main__':main()
