"""Only IO recovery behavior; no simulations, learning or evaluation cases."""
import json
import pytest
from tools.atomic_io_retry import retry_io


def test_retry_keeps_payload_and_completes_after_transient_lock(tmp_path,monkeypatch):
    monkeypatch.setattr('tools.atomic_io_retry.time.sleep',lambda value:None)
    target=tmp_path/'progress.json';calls=[];payload=dict(policy='fixed',cases=5)
    def write(value):
        calls.append(value.copy())
        if len(calls)<3:raise PermissionError('sharing violation fixture')
        target.write_text(json.dumps(value));return 'saved'
    assert retry_io(write,payload)=='saved'
    assert json.loads(target.read_text())==payload and calls==[payload]*3


def test_permanent_lock_is_bounded_and_other_errors_are_not_retried(monkeypatch):
    monkeypatch.setattr('tools.atomic_io_retry.time.sleep',lambda value:None)
    calls=[]
    def locked():calls.append(1);raise PermissionError('permanent fixture')
    with pytest.raises(PermissionError):retry_io(locked,attempts=3)
    assert len(calls)==3
    with pytest.raises(ValueError):retry_io(lambda:(_ for _ in ()).throw(ValueError('not an IO lock')))
