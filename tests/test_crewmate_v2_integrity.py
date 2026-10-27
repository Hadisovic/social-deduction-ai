"""Historical source/selection preservation and independent V2 compatibility contract."""
import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from crewmate_v2.config import LobbySettings
from crewmate_v2.provenance import metadata, validate_metadata
from phase6.config import source_hashes, verify_frozen

BASE='2baa5dbc1b36b047ed97934d69526573b510c280'


def test_every_historical_compatibility_source_matches_verified_base():
    for name,value in source_hashes().items():
        original=subprocess.check_output(['git','show',BASE+':'+name])
        if name.endswith('.py'):
            assert hashlib.sha256(original.replace(b'\r\n',b'\n')).hexdigest()==value,name
        else:
            # Git stores LF, while the original Windows checkpoint pins raw checkout
            # bytes for JSON. Do not rewrite that file or weaken its strict loader.
            assert original.replace(b'\r\n',b'\n')==Path(name).read_bytes().replace(b'\r\n',b'\n'),name
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==value
    verify_frozen()


def test_all_nine_selection_locks_and_checkpoints_remain_exact():
    candidates=json.loads(Path('docs/benchmarks/phase6/remaining-six/final-candidates-proposal.json').read_text())['candidates']
    assert len(candidates)==9
    for c in candidates:
        assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']
    for name in ('docs/benchmarks/phase6/remaining-six/selection.json','docs/benchmarks/phase6/seed17-stage/selection.json'):
        assert Path(name).read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show',BASE+':'+name]).replace(b'\r\n',b'\n')


def test_v2_rejects_historical_schema_and_mismatched_lobby_sources():
    lobby=LobbySettings();record=metadata(lobby)
    assert validate_metadata(record,lobby)
    with pytest.raises(ValueError,match='historical'):validate_metadata({'phase6_sources':source_hashes()},lobby)
    with pytest.raises(ValueError):validate_metadata(record,LobbySettings(visual_tasks=False))
    record=dict(record,sources={})
    with pytest.raises(ValueError):validate_metadata(record,lobby)


def test_existing_strict_loader_still_loads_six_continuation_policies():
    import torch
    from phase6.train import load_policy
    torch.set_num_threads(1)
    candidates=json.loads(Path('docs/benchmarks/phase6/remaining-six/final-candidates-proposal.json').read_text())['candidates']
    for c in candidates:
        if c['initialization_seed']!=17:
            _,experiment,record=load_policy(c['path'])
            assert experiment.memory==c['condition'] and record['initialization_seed']==c['initialization_seed']


def test_new_replay_and_reset_have_v2_schema_independent_streams_and_same_roles():
    from crewmate_v2.runtime import V2StrategicMatch
    from social_deduction.actor import Role
    a=V2StrategicMatch(2610003,lobby=LobbySettings(timeout=1.))
    for _ in range(5):a.tick()
    a.reset(2610004)
    b=V2StrategicMatch(2610004,lobby=LobbySettings(timeout=1.),planner=a.game.planner)
    assert a.focal==b.focal and a.game.players[a.focal].role is Role.CREWMATE
    assert a.game.snapshot()==b.game.snapshot()
    from crewmate_v2 import SCHEMA
    assert a.replay_document()['schema']==SCHEMA+'-replay'
    assert len({id(c) for c in a.controllers.values()})==5
