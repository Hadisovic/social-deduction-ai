"""Exclusive checkpoints, strict source/schema integrity and independent champion."""
from dataclasses import asdict
import hashlib
import json
import random
import platform
from pathlib import Path
import numpy as np
import torch
from . import SCHEMA

ROOT=Path(__file__).resolve().parents[1]
DEPENDENCIES=('phase3_engine.py','among_us_map.py','among_us_map_simulation.py','navigation_service.py',
    'social_deduction/phase3_api.py','social_deduction/phase3_observation.py','social_deduction/actor.py',
    'social_deduction/truth.py','crewmate_v2/config.py','crewmate_v2/engine.py','crewmate_v2/tasks.py',
    'crewmate_v2/ghosts.py','crewmate_v2/api.py','crewmate_v2/phase21/opponents.py',
    'crewmate_v2/phase2/opponents.py','phase5_training.py')


def source_lock():
    paths=[ROOT/p for p in DEPENDENCIES]+list((ROOT/'patrick_jane').glob('*.py'))
    paths+=list((ROOT/'assets/skeld').rglob('*.json'))
    paths+=[ROOT/p for p in ('phase3_renderer.py','among_us_renderer.py')]
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(paths)}


def save(path,policy,optimizer,experiment,counters,*,rng,metadata=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    record=dict(schema=SCHEMA,sources=source_lock(),runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__)),
        architecture=policy.architecture(),experiment=experiment.to_dict(),
        state_dict=policy.state_dict(),optimizer=optimizer.state_dict(),counters=counters,
        rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 numpy=rng.bit_generator.state,python=random.getstate()),metadata=metadata or {},boundary='complete-matches')
    # Never overwrite even a failed or old checkpoint name.
    with path.open('xb') as handle:torch.save(record,handle)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path,policy,optimizer=None,*,rng=None,restore_rng=True):
    record=torch.load(path,map_location='cpu',weights_only=True)
    runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__))
    if record.get('schema')!=SCHEMA or record.get('sources')!=source_lock() or record.get('architecture')!=policy.architecture() or record.get('runtime')!=runtime:
        raise ValueError('Patrick Jane checkpoint schema/source/architecture mismatch')
    policy.load_state_dict(record['state_dict'],strict=True)
    if any(not torch.isfinite(p).all() for p in policy.parameters()):raise ValueError('Nonfinite checkpoint')
    if optimizer is not None:optimizer.load_state_dict(record['optimizer'])
    if restore_rng:
        torch.set_rng_state(record['rng']['torch'])
        if torch.cuda.is_available() and record['rng']['cuda']:torch.cuda.set_rng_state_all(record['rng']['cuda'])
        random.setstate(record['rng']['python'])
        if rng is not None:rng.bit_generator.state=record['rng']['numpy']
    return record


def select_champion(checkpoint,matches,folder):
    """Comparable held-out games only; no training rollback and no reward ranking."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);pointer=folder/'champion.json'
    score=dict(wins=sum(m['winner']=='crewmate' for m in matches),games=len(matches),
               task_wins=sum(m['reason']=='TASKS_COMPLETED' for m in matches),
               correct_ejections=sum(m['correct_ejections'] for m in matches),
               innocent_ejections=sum(m['innocent_ejections'] for m in matches))
    if not matches:raise ValueError('Champion requires held-out evaluation')
    previous=json.loads(pointer.read_text()) if pointer.exists() else None
    signature=[(m['seed'],m['players'],m['impostors']) for m in matches]
    if previous and previous['scenario_signature']!=[list(x) for x in signature]:
        raise ValueError('Champion comparisons require identical held-out scenarios/population')
    new_key=(score['wins']/score['games'],score['task_wins'],score['correct_ejections']-score['innocent_ejections'])
    old_key=tuple(previous['selection_key']) if previous else None
    if old_key is None or new_key>old_key:
        source=Path(checkpoint);destination=folder/source.name
        with destination.open('xb') as handle:handle.write(source.read_bytes())
        record=dict(checkpoint=destination.name,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                    score=score,selection_key=new_key,scenario_signature=signature)
        temporary=folder/'champion.json.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(pointer)
        return True
    return False
