"""Locked final evaluation, scientific plots and identity/evidence audits."""
import json
from pathlib import Path
import numpy as np
import torch
from belief.features import FEATURE_NAMES, INDEX, probabilities, rule_logits
from belief.metrics import match_weights, metrics, paired_nll_interval
from belief.model import BeliefModel, batch_logits
from phase4_data import load_split


def evidence_probes(model):
    inputs = np.zeros((6, 4, len(FEATURE_NAMES)), dtype=np.float32)
    names = ['no_evidence', 'direct_elimination', 'claimed_elimination',
             'direct_contradiction', 'claim_conflict', 'direct_near_body']
    for i, name in enumerate(names[1:], 1):
        inputs[i, 0, INDEX[name]] = 1
    logits = batch_logits(model, inputs)
    p = probabilities(logits, model.temperature)
    perm = [2, 0, 3, 1]
    permuted = probabilities(batch_logits(model, inputs[:, perm]), model.temperature)
    return {'scenarios': {n: {'probabilities': list(map(float, row)),
                              'entropy': float(-(row*np.log(row)).sum())}
                          for n,row in zip(names,p)},
            'permutation_max_error': float(np.abs(permuted-p[:,perm]).max()),
            'notice': 'Controlled feature probes, not empirical real-match accuracy; all other evidence zero.'}


def identity_audit(prob, data, directory, split):
    records = {int(p.stem): json.loads(p.read_text()) for p in (Path(directory)/split).glob('*.json')}
    weights = match_weights(data['match'])
    accum = {kind: {} for kind in ('public_id', 'color', 'spawn')}
    for row, p in enumerate(prob):
        record = records[int(data['match'][row])]
        for column, pid in enumerate(data['candidates'][row]):
            keys = {'public_id': str(pid), 'color': record['colors'][pid],
                    'spawn': str(record['spawns'][pid])}
            for kind, key in keys.items():
                entry = accum[kind].setdefault(key, [0., 0., 0.])
                entry[0] += weights[row]
                entry[1] += weights[row] * p[column]
                entry[2] += weights[row] * float(data['y'][row] == column)
    return {kind: {key: {'candidate_mass': v[0], 'mean_probability_when_candidate': v[1]/v[0],
                         'empirical_role_rate_when_candidate': v[2]/v[0]}
                    for key,v in entries.items()} for kind,entries in accum.items()}


def evaluate(data_dir, model_dir, report_dir):
    torch.set_num_threads(1)
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads((model_dir/'selection.json').read_text())
    model = BeliefModel.load(model_dir/'belief.pt')
    result = {'metric_weighting': 'equal match weight, equal sample weight within each match',
              'accuracy': 'fractional correctness for exactly tied maximum probabilities',
              'brier': 'sum over four classes (uniform prior .75)',
              'ece': '10 equal-width top-confidence bins', 'splits': {},
              'evidence_probes': evidence_probes(model), 'selection': {
                  'architecture': selection['selected_architecture'], 'seed': selection['selected_seed'],
                  'temperature': model.temperature, 'rule_temperature': selection['rule_temperature']}}
    plot_data = {}
    for split in ('test', 'heldout'):
        data = load_split(data_dir, split)
        raw = batch_logits(model, data['x_full'])
        rules = rule_logits(data['x_full'])
        predictions = {'prior': np.full((len(data['y']),4), .25), 'rules_raw': probabilities(rules),
            'rules_calibrated': probabilities(rules, selection['rule_temperature']),
            'learned_raw': probabilities(raw), 'learned_calibrated': probabilities(raw, model.temperature)}
        for record in selection['ablations']:
            ablation = BeliefModel.load(model_dir/record['checkpoint'])
            predictions[record['view']] = probabilities(batch_logits(ablation, data['x_'+record['view']]), ablation.temperature)
        scores = {name: metrics(p, data['y'], data['match']) for name,p in predictions.items()}
        variations = []
        for record in selection['candidates']:
            if record['architecture'] == selection['selected_architecture']:
                candidate = BeliefModel.load(model_dir/record['checkpoint'])
                p = probabilities(batch_logits(candidate, data['x_full']), candidate.temperature)
                variations.append({'seed': record['seed'], 'metrics': metrics(p,data['y'],data['match'])})
        groups = {'early_0_12s': data['tick']*.2 < 12, 'middle_12_36s': (data['tick']*.2 >= 12)&(data['tick']*.2 < 36),
                  'late_36s_plus': data['tick']*.2 >= 36}
        groups.update({stage: data['stage'] == stage for stage in np.unique(data['stage'])})
        stages = {key: {name: metrics(p[mask],data['y'][mask],data['match'][mask])
                        for name,p in predictions.items()} for key,mask in groups.items() if mask.any()}
        result['splits'][split] = {'models': scores, 'stages': stages, 'selected_architecture_seeds': variations,
            'learned_minus_calibrated_rules_nll': paired_nll_interval(predictions['learned_calibrated'],
                predictions['rules_calibrated'], data['y'],data['match']),
            'identities': identity_audit(predictions['learned_calibrated'],data,data_dir,split),
            'samples': len(data['y']), 'matches': len(np.unique(data['match']))}
        plot_data[split] = predictions, data
        print(f'\n{split}: {len(np.unique(data["match"]))} matches / {len(data["y"])} samples')
        for name, score in scores.items():
            print(f"{name:22} accuracy={score['accuracy']:.4f} NLL={score['nll']:.4f} Brier={score['brier']:.4f} ECE={score['ece']:.4f}")
    (report_dir/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    plot_results(result, selection, plot_data, report_dir)
    return result


def plot_results(result, selection, plot_data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 140, 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.bbox': 'tight'})
    colors = {'prior':'#8796a5','rules_calibrated':'#b67b2e','learned_raw':'#4379b8','learned_calibrated':'#178879'}
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, split in zip(axes, ('test','heldout')):
        ax.plot([0,1],[0,1], '--', color='.7')
        for name in colors:
            bins = result['splits'][split]['models'][name]['reliability']
            valid = [b for b in bins if b['mass']]
            ax.plot([b['confidence'] for b in valid], [b['accuracy'] for b in valid], 'o-',label=name,color=colors[name])
        ax.set(xlabel='Mean confidence',ylabel='Empirical correctness',title=split, xlim=(0,1),ylim=(0,1))
    axes[-1].legend(fontsize=8); fig.suptitle('Reliability • match-balanced • 10 equal-width bins')
    fig.savefig(output/'reliability.png'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, split in zip(axes, ('test','heldout')):
        predictions,data = plot_data[split]
        for name in ('learned_raw','learned_calibrated'):
            ax.hist(predictions[name].max(1), bins=np.linspace(.25,1,16), weights=match_weights(data['match']),
                    histtype='step',linewidth=2,label=name,color=colors[name])
        ax.set(title=split,xlabel='Top probability',ylabel='Match-balanced mass'); ax.legend(fontsize=8)
    fig.savefig(output/'confidence.png'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for r in selection['candidates']:
        axes[0].plot([e['epoch'] for e in r['history']], [e['validation_nll'] for e in r['history']],
                     label=f"{r['architecture']} / {r['seed']}")
    axes[0].set(xlabel='Epoch',ylabel='Validation NLL',title='Architecture/seed selection'); axes[0].legend(fontsize=8)
    curves = selection['learning_curve']
    axes[1].plot([c['train_matches'] for c in curves],[c['validation_nll'] for c in curves],'o-')
    axes[1].set(xlabel='Training matches',ylabel='Validation NLL',title='Dataset-size learning curve')
    fig.savefig(output/'learning_curves.png'); plt.close(fig)
    names = ['prior','rules_calibrated','learned_calibrated','current','no_claims','collapsed']
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax,split in zip(axes,('test','heldout')):
        ax.barh(names,[result['splits'][split]['models'][n]['nll'] for n in names],color='#397d94')
        ax.invert_yaxis(); ax.set(xlabel='NLL (lower is better)',title=split)
    fig.tight_layout(); fig.savefig(output/'baselines_ablations.png'); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    stage_names = ['early_0_12s','middle_12_36s','late_36s_plus','discussion','voting']
    for ax,split in zip(axes,('test','heldout')):
        stages = result['splits'][split]['stages']
        present = [s for s in stage_names if s in stages]
        for n in ('prior','rules_calibrated','learned_calibrated'):
            ax.plot(range(len(present)),[stages[s][n]['nll'] for s in present],'o-',label=n)
        ax.set_xticks(range(len(present)),present,rotation=35,ha='right'); ax.set(title=split,ylabel='NLL')
    axes[-1].legend(fontsize=8); fig.tight_layout(); fig.savefig(output/'stages.png'); plt.close(fig)
