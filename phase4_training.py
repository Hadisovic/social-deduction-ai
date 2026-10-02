"""Trusted supervised optimization; validation selection precedes all final tests."""
from copy import deepcopy
import json
from pathlib import Path
import platform
import random
from time import perf_counter
import numpy as np
import torch
from torch import nn

from belief.features import rule_logits
from belief.metrics import fit_temperature, match_weights
from belief.model import BeliefModel, CandidateNet, batch_logits
from phase4_data import load_split


def train_one(train, validation, architecture, seed, view='full', epochs=160, patience=20):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model = CandidateNet(architecture)
    x = torch.from_numpy(train['x_'+view]); y = torch.from_numpy(train['y'])
    weights = torch.from_numpy(match_weights(train['match']).astype(np.float32) * len(y))
    with torch.no_grad():
        model.mean.copy_(x.mean((0, 1)))
        model.scale.copy_(x.std((0, 1)).clamp(min=.1))
    vx = torch.from_numpy(validation['x_'+view]); vy = torch.from_numpy(validation['y'])
    vw = torch.from_numpy(match_weights(validation['match']).astype(np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.001)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, state = float('inf'), 0, None
    history = []
    started = perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        total = 0.
        for ids in order.split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = (nn.functional.cross_entropy(model(x[ids]), y[ids], reduction='none') * weights[ids]).mean()
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(ids) / len(y)
        model.eval()
        with torch.inference_mode():
            validation_loss = float((nn.functional.cross_entropy(model(vx), vy, reduction='none') * vw).sum())
        history.append({'epoch': epoch+1, 'train_nll': total, 'validation_nll': validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, state = validation_loss, epoch+1, deepcopy(model.state_dict())
        if epoch+1-best_epoch >= patience:
            break
    model.load_state_dict(state)
    result = BeliefModel(model, view=view)
    validation_logits = batch_logits(result, validation['x_'+view])
    result.temperature = fit_temperature(validation_logits, validation['y'], validation['match'])
    record = {'architecture': architecture, 'seed': seed, 'view': view,
              'parameters': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch,
              'validation_nll': best_loss, 'temperature': result.temperature, 'history': history,
              'wall_seconds': perf_counter()-started}
    return result, record


def train_experiment(data_dir, output_dir, quick=False):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    train = load_split(data_dir, 'train'); validation = load_split(data_dir, 'validation')
    assert not set(train['match']) & set(validation['match'])
    records = []
    seeds = (17,) if quick else (17, 29, 43)
    epochs = 12 if quick else 160
    for architecture in ('linear', 'set'):
        for seed in seeds:
            model, record = train_one(train, validation, architecture, seed, epochs=epochs)
            name = f'{architecture}-{seed}.pt'
            model.save(output / name, {'training': {k:v for k,v in record.items() if k != 'history'}})
            record['checkpoint'] = name
            records.append(record)
            print(f"Trained {name}: validation NLL {record['validation_nll']:.4f}, T={model.temperature:.3f}", flush=True)
    architecture_scores = {a: float(np.mean([r['validation_nll'] for r in records if r['architecture'] == a]))
                           for a in ('linear', 'set')}
    selected_arch = min(architecture_scores, key=architecture_scores.get)
    selected = min((r for r in records if r['architecture'] == selected_arch), key=lambda r:r['validation_nll'])
    model = BeliefModel.load(output / selected['checkpoint'])
    model.save(output / 'belief.pt', {'selection': {k:v for k,v in selected.items() if k != 'history'}})
    ablations = []
    for view in ('current', 'no_claims', 'collapsed'):
        ablation, record = train_one(train, validation, selected_arch, 17, view, epochs=epochs)
        record['checkpoint'] = f'{view}.pt'
        ablation.save(output / record['checkpoint'])
        ablations.append(record)
        print(f"Ablation {view}: validation NLL {record['validation_nll']:.4f}", flush=True)
    curves = []
    if not quick:
        for count in (200, 400):
            chosen = np.unique(train['match'])[:count]
            mask = np.isin(train['match'], chosen)
            subset = {k:v[mask] for k,v in train.items()}
            _, record = train_one(subset, validation, selected_arch, 17)
            curves.append({'train_matches': count, 'validation_nll': record['validation_nll']})
        full_seed17 = next(r for r in records if r['architecture'] == selected_arch and r['seed'] == 17)
        curves.append({'train_matches': 800, 'validation_nll': full_seed17['validation_nll']})
    rule_temperature = fit_temperature(rule_logits(validation['x_full']), validation['y'], validation['match'])
    selection = {'selected_architecture': selected_arch, 'selected_seed': selected['seed'],
        'architecture_mean_validation_nll': architecture_scores, 'candidates': records,
        'ablations': ablations, 'learning_curve': curves, 'rule_temperature': rule_temperature,
        'selection_criterion': 'mean validation NLL across seeds, then best validation seed',
        'training': {'optimizer': 'Adam', 'lr': .003, 'weight_decay': .001, 'batch_size': 512,
            'max_epochs': epochs, 'early_stopping_patience': 20, 'loss': 'match-balanced cross entropy',
            'seeds': list(seeds), 'device': 'cpu', 'torch_threads': 1, 'deterministic_algorithms': True},
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'numpy': np.__version__, 'platform': platform.platform()},
        'final_tests_consulted': False, 'quick': quick}
    (output / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print('Selection and calibration locked. Final tests may now be evaluated.', flush=True)
    return selection
