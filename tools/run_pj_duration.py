"""Spawn-safe manual CLI. No Torch imports in CPU worker bootstrap."""
import argparse,json,os,sys,traceback
from pathlib import Path


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser();p.add_argument('--folder',default='results/task-pursuit-v1');p.add_argument('--origin',default='origin')
    p.add_argument('--hours',type=float,default=12.);p.add_argument('--setup-check',action='store_true')
    a=p.parse_args()
    if a.setup_check:
        import torch,numpy as np
        from patrick_jane.pursuit_transition import import_update18
        from patrick_jane.network import PatrickJane
        if not torch.cuda.is_available():raise RuntimeError('NVIDIA CUDA unavailable. Install a GTX1650-compatible driver supporting the bundled CUDA12.8 runtime; CPU fallback is disabled.')
        torch.ones((2,2),device='cuda') @ torch.ones((2,2),device='cuda');torch.cuda.synchronize()
        policy=PatrickJane();optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
        counters,provenance=import_update18(a.origin,policy,optimizer,np.random.default_rng(17))
        from patrick_jane.duration_run import experiment
        root=Path(__file__).resolve().parents[1];config=json.loads((root/'portable-config.json').read_text())
        if config['workers']!=6 or config['experiment']!=json.loads(json.dumps(experiment().to_dict())):raise ValueError('Packaged configuration does not match implemented profile')
        print(json.dumps(dict(runtime_ready=True,gpu=torch.cuda.get_device_name(0),python=sys.version,torch=str(torch.__version__),numpy=np.__version__,lifetime_decisions=counters['individual_decisions'],import_provenance=provenance),indent=2));return 0
    from patrick_jane.learning_run import ROOT
    from patrick_jane.duration_run import train
    from tools.run_pj_learning import Tee
    folder=Path(a.folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    with (folder/'console.log').open('a',encoding='utf-8',buffering=1) as log:
        out,err=sys.stdout,sys.stderr;sys.stdout=Tee(out,log);sys.stderr=Tee(err,log)
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # Only prevent idle sleep for this process lifetime.
        try:return train(folder,Path(a.origin).resolve(),hours=a.hours)
        except BaseException:traceback.print_exc();print('STARTUP/TRAINING FAILED; existing checkpoints preserved',flush=True);return 1
        finally:
            if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            sys.stdout=out;sys.stderr=err

if __name__=='__mp_main__' and 'torch' in sys.modules:raise RuntimeError('CPU worker imported Torch')
if __name__=='__main__':raise SystemExit(main())
