"""Windows spawn-safe A2 launcher; no automatic long-run invocation on import."""
import argparse,json,os,sys,traceback
from pathlib import Path


def choose_workers(folder,requested):
    if (folder/'registration.json').exists():
        frozen=json.loads((folder/'registration.json').read_text())['workers']
        if requested and requested!=frozen:raise ValueError('Cannot change workers while resuming')
        return frozen
    import psutil
    recommended=min(6,psutil.cpu_count(logical=False) or 2,max(1,int(psutil.virtual_memory().total/(2*1024**3))))
    if requested and not 1<=requested<=recommended:raise ValueError(f'Workers must fit detected CPU/RAM; recommended maximum {recommended}')
    return requested or recommended


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    parser=argparse.ArgumentParser();parser.add_argument('--folder',default='results/investigation-a2-v1');parser.add_argument('--origin',default='origin')
    parser.add_argument('--hours',type=float,default=12.);parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--additional-decisions',type=int);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--setup-check',action='store_true')
    parser.add_argument('--reward-regime',choices=['investigation-a2-v1','task-pursuit-v1'],default='investigation-a2-v1')
    parser.add_argument('--opponent-regime',choices=['historical','upgraded-a2-v1','mixed-a2-v1'],default='mixed-a2-v1')
    a=parser.parse_args();folder=Path(a.folder).resolve()
    import torch,numpy as np
    from patrick_jane.a2_transition import import_update102
    from patrick_jane.network import PatrickJane
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable: install a GTX1650-compatible NVIDIA driver for bundled CUDA12.8. No CPU fallback.')
    workers=choose_workers(folder,a.workers)
    if a.setup_check:
        torch.ones((2,2),device='cuda')@torch.ones((2,2),device='cuda');torch.cuda.synchronize()
        p=PatrickJane();optimizer=torch.optim.Adam(p.parameters(),lr=.0003)
        c,provenance=import_update102(a.origin,p,optimizer,np.random.default_rng(17))
        print(json.dumps(dict(runtime_ready=True,workers=workers,gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),lifetime_decisions=c['individual_decisions'],provenance=provenance),indent=2));return 0
    from patrick_jane.a2_run import train
    from tools.run_pj_learning import Tee
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'console.log').open('a',encoding='utf-8',buffering=1) as log:
        out,err=sys.stdout,sys.stderr;sys.stdout=Tee(out,log);sys.stderr=Tee(err,log)
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        from patrick_jane.a2_config import A2Experiment
        profile=A2Experiment(reward_regime=a.reward_regime,living_task_reward=.5 if a.reward_regime=='investigation-a2-v1' else 2.,opponent_regime=a.opponent_regime)
        try:return train(folder,a.origin,hours=a.hours,workers=workers,additional_decisions=a.additional_decisions,prepare_only=a.prepare_only,profile=profile)
        except BaseException:traceback.print_exc();print('A2 FAILED; previous checkpoints preserved',flush=True);return 1
        finally:
            if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            sys.stdout=out;sys.stderr=err

if __name__=='__mp_main__' and 'torch' in sys.modules:raise RuntimeError('CPU worker imported Torch')
if __name__=='__main__':raise SystemExit(main())
