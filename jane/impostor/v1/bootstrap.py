"""Verified external engine dependency, not a vendored crew network."""
import hashlib
import json
from pathlib import Path
import sys


def activate(source):
    from jane.shared.patrick_jane import reference_review
    root = Path(source).resolve()
    receipt = reference_review()
    for name, expected in {**receipt['source_hashes'], **receipt['config_import_source_hashes']}.items():
        actual = hashlib.sha256((root / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected:
            raise ValueError('Upstream source mismatch: ' + name)
    for name, expected in receipt['published_files_verified'].items():
        path = root / 'artifacts/patrick-jane/task-pursuit-v1-update102' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('Upstream artifact mismatch: ' + name)
    for name in ('patrick_jane', 'crewmate_v2', 'social_deduction', 'phase3_engine', 'spectator'):
        module = sys.modules.get(name)
        if module and not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError('Refusing mixed source roots: ' + name)
    sys.path.insert(0, str(root))
    return root


def experiment(profile):
    from jane.shared.settings import load_settings
    from jane.shared.patrick_jane import a2_configuration
    from patrick_jane.config import Population, ClassicLobby, SabotageSettings
    from patrick_jane.a2_config import A2Experiment
    settings = load_settings(profile)
    value = a2_configuration(settings)
    value['population'] = Population(**value['population'])
    value['lobby'] = ClassicLobby(**value['lobby'])
    value['sabotage'] = SabotageSettings(**value['sabotage'])
    value['allowed_sabotages'] = tuple(value['allowed_sabotages'])
    return settings, A2Experiment(**value)


def frozen_crew(root):
    import torch
    if str(torch.__version__).split('+')[0] != '2.7.1':
        raise ValueError('Frozen published crew inference requires the explicitly supported Torch 2.7.1 runtime')
    from patrick_jane.network import PatrickJane
    path = root / 'artifacts/patrick-jane/task-pursuit-v1-update102/inference.pt'
    with torch.random.fork_rng(devices=[]):
        model = PatrickJane()
    record = torch.load(path, map_location='cpu', weights_only=True)
    if record['architecture'] != model.architecture():
        raise ValueError('Published crew architecture mismatch')
    model.load_state_dict(record['state_dict'], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def source_lock():
    folder = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted(folder.glob('*.py'))}
