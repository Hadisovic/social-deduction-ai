"""Explicit new partitions and provenance; no training/final-evaluation driver."""
import hashlib
import platform
import os
import subprocess
from dataclasses import asdict
import numpy as np
import torch
from phase6.config import ROOT, source_hashes
from . import SCHEMA
from .frozen import CHECKPOINT, SHA256, ADAPTER

BASE = 'e62ba0b6c33ffd0317f78e18b4289d778cd5af2b'
PARTITIONS = dict(train=(3100000,3199999),validation=(3200000,3200199),
                  final_reserved=(3300000,3300499),development=(3500000,3500019))


def development_seeds(count):
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError('Bounded development supports 1..20 scenarios')
    return tuple(range(PARTITIONS['development'][0],PARTITIONS['development'][0]+count))


def manifest(env):
    paths = sorted([*ROOT.glob('impostor_research/*.py'),*ROOT.glob('crewmate_v2/*.py'),
                    ROOT/'configs/crewmate-v2-lobby.json',ROOT/'tools/benchmark_impostor_v2.py'])
    sources = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths}
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    return dict(schema=SCHEMA,baseline=BASE,source_commit=git('rev-parse','HEAD'),
        branch=git('branch','--show-current'),dirty=bool(git('status','--porcelain')),
        sources=sources,historical_sources=source_hashes(),lobby=env.lobby.to_dict(),
        mixture=asdict(env.mixture),safeguards=asdict(env.safeguards),reward=asdict(env.reward_config),
        adapter=ADAPTER,checkpoint=CHECKPOINT,checkpoint_sha256=SHA256,frozen_tensors=env.models.hashes(),
        execution=dict(fast=env.fast,coalesce=env.coalesce,dt=env.lobby.dt,decision_interval=env.lobby.decision_interval),
        partitions=PARTITIONS,versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        compute=dict(processor=platform.processor(),logical_cpus=os.cpu_count(),torch_threads=torch.get_num_threads(),
                     device='cpu',workers=1,os=platform.platform()),
        training_authorized=False,engineering_only=True)
