"""Durable active-time budget; no simulator, Torch or wall-clock deadline policy."""
from pathlib import Path
from time import perf_counter
from .progress import atomic_json
from .learning_report import read


class ActiveBudget:
    def __init__(self,folder,hours,clock=perf_counter):
        self.folder=Path(folder);self.clock=clock;self.started=None
        self.seconds=float(read(self.folder/'active-time.json',{}).get('seconds',0))
        journal=read(self.folder/'duration-budget.json',[])
        requested=hours*3600
        if journal and requested<journal[-1]['target_seconds']:raise ValueError('Duration can only be retained or extended')
        if not journal or requested>journal[-1]['target_seconds']:
            journal.append(dict(target_seconds=requested,active_seconds_at_extension=self.seconds))
            atomic_json(self.folder/'duration-budget.json',journal)
        self.target=requested

    def value(self):return self.seconds+(self.clock()-self.started if self.started is not None else 0.)
    def persist(self):atomic_json(self.folder/'active-time.json',dict(seconds=self.value(),target_seconds=self.target,in_flight=self.started is not None))
    def begin(self):
        if self.started is not None:raise RuntimeError('Nested training clock')
        self.started=self.clock();self.persist()
    def end(self):
        if self.started is not None:self.seconds=self.value();self.started=None
        self.persist()
    @property
    def reached(self):return self.value()>=self.target
    def telemetry(self):
        seconds=self.value()
        return dict(active_training_seconds=seconds,target_training_seconds=self.target,
            progress_percent=min(100.,100*seconds/self.target),eta_seconds=max(0.,self.target-seconds),
            finishing_boundary=self.reached and self.started is not None)
