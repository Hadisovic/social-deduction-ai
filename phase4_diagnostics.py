"""Validation-only negative control and exact training reproduction. No test tuning."""
import json
from pathlib import Path
import numpy as np
import torch
from belief.features import probabilities
from belief.metrics import metrics
from belief.model import BeliefModel, batch_logits
from phase4_data import load_split
from phase4_training import train_one


def run_diagnostics(data_dir, model_dir, report_dir):
    train = load_split(data_dir,'train'); validation = load_split(data_dir,'validation')
    selected = json.loads((Path(model_dir)/'selection.json').read_text())
    original = BeliefModel.load(Path(model_dir)/'belief.pt')
    repeated, record = train_one(train,validation,selected['selected_architecture'],selected['selected_seed'])
    exact = all(torch.equal(value,repeated.network.state_dict()[key]) for key,value in original.network.state_dict().items())
    assert exact and repeated.temperature == original.temperature
    # Randomize one target per match/focal history, preserving temporal correlation.
    # Validation targets remain correct. This is solely a negative control.
    shuffled = dict(train)
    rng = np.random.default_rng(8675309)
    lookup = {(int(g),str(a)):int(rng.integers(4)) for g,a in sorted(set(zip(train['match'],train['actor'])))}
    shuffled['y'] = np.array([lookup[(int(g),str(a))] for g,a in zip(train['match'],train['actor'])],dtype=np.int64)
    negative, negative_record = train_one(shuffled,validation,selected['selected_architecture'],17)
    p = probabilities(batch_logits(negative,validation['x_full']))
    report = {'exact_retraining_state_dict':exact,'exact_retraining_temperature':True,
              'retraining_best_epoch':record['best_epoch'], 'negative_control':{
                  'method':'Random categorical target per match/focal training history; true validation labels',
                  'seed':8675309,'validation_raw_metrics':metrics(p,validation['y'],validation['match']),
                  'best_epoch':negative_record['best_epoch']},
              'notice':'Diagnostic only. Release selection unchanged; final tests are not used by this script.'}
    Path(report_dir).mkdir(parents=True,exist_ok=True)
    (Path(report_dir)/'reproducibility.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    run_diagnostics(root/'artifacts/phase4/data',root/'artifacts/phase4',root/'docs/benchmarks/phase4')
