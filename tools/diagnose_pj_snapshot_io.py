"""Frozen development-only validation batch with controlled observer read locks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import json
import threading
import time


def main():
    import torch
    from patrick_jane.config import Experiment,DEV_VALIDATION
    from patrick_jane.network import PatrickJane
    from patrick_jane.collector import Pool,collect
    from patrick_jane.health import audit_rollout
    from patrick_jane.checkpoint import source_lock,freeze_sources
    from patrick_jane.progress import atomic_json,Progress
    from patrick_jane.telemetry import Resources
    folder=Path('local-runs/patrick-jane/snapshot-io-diagnostic-v1').resolve();folder.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    policy=PatrickJane().cuda();experiment=Experiment();sources=source_lock()
    atomic_json(folder/'registration.json',dict(development_only=True,training_updates=0,seeds=list(DEV_VALIDATION[32:64]),
        sources=sources,source_snapshot=freeze_sources(folder),reader_lock_seconds=.010,reader_interval_seconds=.030,
        purpose='Reproduce the failed frozen-validation batch with full summaries and observer lock stress'))
    stop=threading.Event();reads=[]
    def reader():
        path=folder/'live/env-00.json'
        while not stop.is_set():
            try:
                with path.open('r') as stream:
                    stream.read(1);time.sleep(.010);reads.append(1)
            except OSError:pass
            stop.wait(.030)
    thread=threading.Thread(target=reader,daemon=True);thread.start()
    try:
        with Resources() as resources,Pool(32) as pool:
            progress=Progress(folder,32,resources,mirror='local-runs/patrick-jane/live/status.json')
            progress.stage='diagnostic · frozen validation · no PPO'
            result=collect(policy,pool,experiment,DEV_VALIDATION[32:64],device='cuda',held_out=True,wall_limit=900,
                replay_dir=folder/'replays',record_every=1,monitor_dir=progress.monitor_dir,status_callback=progress.publish)
            result.pop('trajectories');result['reader_openings']=len(reads)
            atomic_json(folder/'metrics.json',result)  # Preserve every outcome before the audit.
            audit_rollout(result)
            if sources!=source_lock():raise RuntimeError('Diagnostic source changed')
            atomic_json(folder/'result.json',dict(passed=True,games=len(result['matches']),reader_openings=len(reads),
                transient_failures=sum(m['monitor_transient_write_failures'] for m in result['matches']),
                illegal_actions=sum(m['metrics']['illegal_actions'] for m in result['matches']),
                monitor_errors=[m['monitor_error'] for m in result['matches'] if m['monitor_error']],training_updates=0))
            progress.stage='diagnostic complete';progress.publish(complete=True)
            print((folder/'result.json').read_text(),flush=True)
    except BaseException as exc:
        atomic_json(folder/'failure.json',dict(error=repr(exc),training_updates=0));raise
    finally:stop.set();thread.join(1.)


if __name__=='__main__':main()
