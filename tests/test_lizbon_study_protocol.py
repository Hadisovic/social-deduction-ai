import importlib.util
from pathlib import Path
import json
import os
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'tools/run_lizbon_classic_study.py'
    spec=importlib.util.spec_from_file_location('lizbon_study',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def row(win,kills=1):
    return dict(winner='impostor' if win else 'crewmate',truncated=False,
        metrics={'eliminations':kills},simulation_seconds=100)


def test_selection_uses_only_complete_validation_and_keeps_reference_on_tie():
    m=module()
    assert m.select({'reference':[row(False)],'stream101':[row(False)]})=='reference'
    assert m.select({'reference':[row(False,3)],'stream101':[row(True,1)]})=='stream101'
    with pytest.raises(ValueError):m.select({'reference':[dict(row(False),truncated=True)]})


def test_selection_lock_is_exclusive_and_hashes_selected_checkpoint(tmp_path):
    m=module();checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'immutable-weights')
    path=tmp_path/'lock.json'
    result=m.lock_selection(path,'reference',checkpoint,{'reference':[row(True)]})
    assert len(result['checkpoint_sha256'])==64
    assert json.loads(path.read_text())['selected']=='reference'
    with pytest.raises(FileExistsError):m.lock_selection(path,'other',checkpoint,{})


def test_fatal_stop_terminates_real_sibling_and_blocks_queued_work():
    import subprocess
    import sys
    m=module();batch=m.ProcessGroup()
    with open(os.devnull,'w') as console:
        slow=batch.start('slow',[sys.executable,'-c','import time; time.sleep(30)'],console)
        bad=batch.start('bad',[sys.executable,'-c','raise SystemExit(1)'],console)
        assert bad.wait(timeout=5)==1
        batch.abort()
        assert slow.wait(timeout=5)!=0
        assert batch.start('queued',[sys.executable,'-c','print(1)'],console) is None
