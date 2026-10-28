"""Wall/resource instrumentation only; no access to simulation random streams."""
import os
import subprocess
import threading
import time


class Resources:
    def __init__(self):self.samples=[];self.stop_event=threading.Event();self.thread=None
    def __enter__(self):
        self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();return self
    def sample(self):
        import psutil
        process=psutil.Process();psutil.cpu_percent()
        while not self.stop_event.is_set():
            try:
                children=process.children(recursive=True)
                item=dict(time=time.time(),cpu_percent=psutil.cpu_percent(),ram_used_bytes=psutil.virtual_memory().used,
                    training_tree_rss_bytes=sum(p.memory_info().rss for p in [process,*children] if p.is_running()))
                command=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],
                    capture_output=True,text=True,timeout=3.,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                if command.returncode==0:
                    util,memory=command.stdout.strip().splitlines()[0].split(',');item.update(gpu_percent=float(util),gpu_memory_mib=float(memory))
                self.samples.append(item)
            except (OSError,ValueError,subprocess.SubprocessError):pass
            self.stop_event.wait(2.)
    def summary(self):
        keys=('cpu_percent','ram_used_bytes','training_tree_rss_bytes','gpu_percent','gpu_memory_mib')
        return {key:dict(mean=sum(x[key] for x in self.samples if key in x)/sum(key in x for x in self.samples),
                         peak=max(x[key] for x in self.samples if key in x)) for key in keys if any(key in x for x in self.samples)}
    def __exit__(self,*args):
        self.stop_event.set()
        if self.thread:self.thread.join(4.)
