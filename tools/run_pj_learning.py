"""Lightweight Windows spawn-safe CLI; imports Torch only in the parent main."""
import argparse,json,os,sys,traceback
from pathlib import Path


class Tee:
    def __init__(self,console,log):self.console=console;self.log=log
    def write(self,value):self.log.write(value);self.log.flush();return self.console.write(value)
    def flush(self):self.console.flush();self.log.flush()
    def isatty(self):return self.console.isatty()
    @property
    def encoding(self):return 'utf-8'


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',required=True);p.add_argument('--additional-decisions',type=int,default=1000000)
    p.add_argument('--workers',type=int,default=32);p.add_argument('--wall-hours',type=float,default=12.)
    p.add_argument('--check-only',action='store_true');p.add_argument('--engineering',action='store_true')
    p.add_argument('--stop-after-updates',type=int);p.add_argument('--finalize',choices=('failed','interrupted'))
    p.add_argument('--reward-v2',action='store_true');p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();folder=Path(args.folder).resolve()
    if args.finalize:
        from patrick_jane.learning_report import read,generate
        from patrick_jane.progress import atomic_json
        from patrick_jane.run_files import run_lock
        if not folder.exists():return 1
        with run_lock(folder):
            if read(folder/'completion-status.json',{}).get('state')=='complete':return 0
            # Progress.elapsed already includes earlier attempts. Do not add
            # active_seconds_before_attempt again after a resumed hard exit.
            status=read(folder/'status.json',{});elapsed=status.get('elapsed',0)
            result=generate(folder,args.finalize,active_seconds=elapsed,error='Foreground process exited without a completed finalizer')
            atomic_json(folder/'completion-status.json',dict(state=args.finalize,success=False,additional_decisions=result['additional_decisions'],active_seconds=elapsed,report=str(folder/'TRAINING_REPORT.md')))
            status.update(stage=args.finalize,training_state=args.finalize,complete=False);atomic_json(folder/'status.json',status)
            print('TRAINING '+args.finalize.upper());print('REPORT: '+str(folder/'TRAINING_REPORT.md'))
        return 1
    from patrick_jane.learning_run import inspect,train
    existing=folder/'registration.json'
    if existing.exists() and json.loads(existing.read_text()).get('schema')!='patrick-jane-lights-run-v1':
        print('TRAINING FAILED — refusing to write into a different historical experiment folder');return 1
    if args.check_only:
        info=inspect(reward_v2=args.reward_v2);info.update(additional_target=args.additional_decisions,workers=args.workers,
            wall_hours=args.wall_hours,run_folder=str(folder),engineering_check=args.engineering)
        info['effective_normal_match_timeout_seconds']=None
        if args.engineering:info['profile']['technical_max_seconds']=12.;info['profile']['evaluation_matches']=2
        print(json.dumps(info,indent=2));return 0
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'console.log').open('a',encoding='utf-8',buffering=1) as log:
        out,err=sys.stdout,sys.stderr;sys.stdout=Tee(out,log);sys.stderr=Tee(err,log)
        try:
            print('RUN FOLDER: '+str(folder),flush=True)
            return train(folder,additional_target=args.additional_decisions,workers=args.workers,
                wall_seconds=args.wall_hours*3600,engineering=args.engineering,stop_after_updates=args.stop_after_updates,reward_v2=args.reward_v2,prepare_only=args.prepare_only)
        except BaseException:
            traceback.print_exc();print('TRAINING FAILED — no successful completion recorded',flush=True);return 1
        finally:sys.stdout=out;sys.stderr=err


if __name__=='__mp_main__' and 'torch' in sys.modules:
    raise RuntimeError('CPU worker bootstrap imported Torch')
if __name__=='__main__':raise SystemExit(main())
