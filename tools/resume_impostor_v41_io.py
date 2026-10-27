"""Resume interrupted evaluation with logging-only sharing-violation retries.

No source/weights/seeds/selection changes. Patches are applied at module import so
Windows spawned workers receive the same IO-only wrapper. No extra training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.atomic_io_retry import retry_io
import impostor_options_v41.study as study

if not getattr(study,'_logging_retry_installed',False):
    original_write=study.write_json;original_progress=study.progress
    def write_json(*args,**kwargs):return retry_io(original_write,*args,**kwargs)
    def progress(*args,**kwargs):return retry_io(original_progress,*args,**kwargs)
    study.write_json=write_json;study.progress=progress;study._logging_retry_installed=True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT):parser.error('Output must remain in project')
    manifest=json.loads((output/'study.json').read_text());lock=json.loads((output/'selection-lock.json').read_text())
    if time()>=manifest['deadline_epoch']:parser.error('Original deadline already reached')
    before=json.loads((output/'progress.json').read_text())
    if before.get('exception')!='PermissionError':parser.error('This recovery is only for the recorded IO failure')
    if not all((output/f'seed-{s}/training.json').exists() for s in (71,83)):
        parser.error('Both training runs must already be complete; this recovery cannot start training')
    recovery=dict(kind='logging_only_resume_no_new_training',started_epoch=time(),failure=before,
        original_deadline_epoch=manifest['deadline_epoch'],selected_checkpoint_sha256=lock['selected']['sha256'],
        selection_lock_sha256=hashlib.sha256((output/'selection-lock.json').read_bytes()).hexdigest(),
        completed_before={p:len(list((output/p).glob('*.json'))) for p in ('selection','diagnostic','confirmation')},
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        helper_sha256=hashlib.sha256((ROOT/'tools/atomic_io_retry.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        algorithms_sources_weights_unchanged=True)
    path=output/f'io-recovery-{int(time())}.json'
    with path.open('x') as stream:json.dump(recovery,stream,indent=2)
    stop=output/'STOP'
    if stop.exists():
        archived=stop.with_name('STOP.io-error-'+str(int(time())))
        if not stop.resolve().is_relative_to(ROOT) or not archived.resolve().is_relative_to(ROOT):raise ValueError('Unsafe marker path')
        stop.rename(archived)  # preserve the failure marker; no deletion
    study.run(output,resume=True)
