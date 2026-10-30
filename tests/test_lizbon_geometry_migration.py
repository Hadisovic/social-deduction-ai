"""Explicit real-checkpoint transfer, never a new reinforcement-learning run."""
import importlib.util
import os
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def migration():
    path=ROOT/'tools/migrate_lizbon_geometry_checkpoint.py'
    spec=importlib.util.spec_from_file_location('geometry_migration',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_unknown_parent_is_rejected_before_model_loading(tmp_path):
    m=migration();parent=tmp_path/'unknown.pt';parent.write_bytes(b'not an approved historical checkpoint')
    with pytest.raises(ValueError,match='registered'):
        m.migrate(parent,tmp_path/'new.pt')
    assert not (tmp_path/'new.pt').exists()


def test_registered_parent_migrates_identical_tensors_with_new_geometry_receipt(tmp_path):
    if not os.environ.get('JANE_UPSTREAM_REVIEW_ROOT'):
        pytest.skip('Run checkpoint integration in the dedicated Jane runtime with explicit source root')
    parent=ROOT/'local-runs/lizbon-classic-v2/study-20261030/train101/checkpoint-003.pt'
    if not parent.exists():pytest.skip('Private historical development checkpoint is not included in fresh clones')
    import torch
    torch.set_num_threads(1)
    from jane.impostor.v1.model import ImpostorPolicy
    from jane.impostor.v1.run import tensor_hash
    from jane.impostor.v2.checkpoint import load
    from jane.classic_v2.settings import load_profile
    profile=load_profile(ROOT/'jane/configs/classic-v2/default_12p.json')
    output=tmp_path/'new.pt';result=migration().migrate(parent,output)
    assert result['tensors_identical'] and result['inherited_optimizer_passes']==24
    model=ImpostorPolicy(731);record=load(output,model,profile)
    assert tensor_hash(model)==result['tensor_sha256']
    assert record['optimizer'] is None and record['counters']['optimizer_steps']==0
    assert record['receipt']['geometry_revision']=='paired-map-planner-v1'
    assert record['ancestry']['transfer'] and record['ancestry']['parent_total_steps']==24
    with pytest.raises(ValueError,match='mismatch'):load(parent,model,profile)
    with pytest.raises(FileExistsError):migration().migrate(parent,output)
    altered=tmp_path/'altered.pt';altered.write_bytes(parent.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='registered'):migration().migrate(altered,tmp_path/'bad.pt')
