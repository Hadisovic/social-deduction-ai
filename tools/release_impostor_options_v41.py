"""Copy all V4.1 study snapshots and the immutable selection into a new bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curate(source,report,bundle):
    source=Path(source).resolve();report=Path(report).resolve();bundle=Path(bundle).resolve()
    study=read(source/'study.json');result=read(source/'results.json');lock=read(source/'selection-lock.json')
    summary=read(report/'summary.json');selected=lock['selected'];release=bundle.parent/'release'
    if result['status']!='complete' or result['final_tests_opened'] or len(result['confirmation'])!=40:
        raise ValueError('Only a completed locked V4.1 study can be curated')
    if selected!=result['selected'] or selected!=summary['selected']:raise ValueError('Selection identity differs')
    if not bundle.is_relative_to(ROOT) or bundle.exists() or release.exists():
        raise ValueError('Choose new in-project bundle/release paths; never overwrite models')
    chosen=ROOT/selected['path']
    if digest(chosen)!=selected['sha256']:raise ValueError('Locked policy hash changed')
    import torch
    from impostor_research.frozen import FrozenModels
    from impostor_options_v41.learning import load
    torch.set_num_threads(1);models=FrozenModels();verified=[]
    for seed in (71,83):
        for step in (0,2048,4096,8192):
            path=source/f'seed-{seed}'/f'checkpoint-{step:05d}.pt';_,metadata=load(path,models)
            if (metadata['seed']!=seed or metadata['decisions']!=step or metadata['development']
                or metadata['schema']!='impostor-options-v41' or 'optimizer' in metadata):
                raise ValueError('Unexpected snapshot identity')
            verified.append((seed,step,path,digest(path)))
    bundle.mkdir(parents=True,exist_ok=False);entries=[]
    for seed,step,path,expected in verified:
        folder=bundle/f'seed-{seed}';folder.mkdir(exist_ok=True);target=folder/path.name
        shutil.copy2(path,target)
        if digest(target)!=expected:raise ValueError('Snapshot copy changed bytes')
        entries.append(dict(seed=seed,decisions=step,path=target.relative_to(ROOT).as_posix(),sha256=expected,strictly_loaded=True))
    release.mkdir(exist_ok=False);target=release/'policy.pt';shutil.copy2(chosen,target)
    _,metadata=load(target,models);models.assert_unchanged()
    if digest(target)!=selected['sha256']:raise ValueError('Selected copy differs')
    metric=summary['confirmation']['selected_v41']
    quality='observed_fresh_wins_small_sample' if metric['wins'] else 'insufficient_winning_strategy'
    manifest=dict(schema=metadata['schema'],source_commit=study['record']['core']['source_commit'],
        selected=selected,checkpoint_sha256=digest(target),selection_lock_sha256=digest(source/'selection-lock.json'),
        confirmation=metric,paired_comparison=summary['paired'],quality=quality,
        report=report.relative_to(ROOT).as_posix(),summary_sha256=digest(report/'summary.json'),
        frozen_weights_unchanged=True,final_tests_opened=False,historical_release_overwritten=False,
        original_deadline_epoch=study['deadline_epoch'],original_study_sha256=study['record']['original_study_sha256'],
        command='python run_impostor_options_v41.py --checkpoint '+target.relative_to(ROOT).as_posix(),
        disclaimer='Immutable best among four preregistered V4.1 candidates by selection; not globally best or final-test accepted.')
    write=lambda path,value:path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write(release/'release.json',manifest);write(report/'release.json',manifest);write(report/'checkpoints.json',entries)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','report','bundle'):parser.add_argument(name,type=Path)
    args=parser.parse_args();print(json.dumps(curate(args.source,args.report,args.bundle),indent=2))
