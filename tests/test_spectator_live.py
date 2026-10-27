"""Recovery fixes: no fabricated progress, robust atomic discovery and real controls."""
import json
import os
from pathlib import Path
import pygame
import pytest
from spectator.replay_format import ReplayRecorder,ReplayHeader,ReplayTick,ReplayResult,ReplayLibrary,parse_replay
from spectator.playback import ReplayPlayer
from spectator.app_state import SpectatorAppState
from spectator.watcher import WatchService
from run_training_spectator import _handle_mouse_click


def make(path,complete=True):
    r=ReplayRecorder(path)
    r.write_header(ReplayHeader(controller_types={'red':'ppo'},config={'training_condition':'Model A — Original Rewards','focal_player':'red','stream':0,'worker':0}))
    r.write_tick(ReplayTick(tick=0,sim_time=0.,players={'red':dict(position=[0.,0.],velocity=[0.,0.],task_progress=.25)}))
    if complete:r.write_result(ReplayResult(tick=1,sim_time=.2,winner='crewmate',reason='TASKS_COMPLETED'))
    else:r.close()
    return r


def test_atomic_no_overwrite_and_live_writing_hidden(tmp_path):
    path=tmp_path/'real.jsonl';r=ReplayRecorder(path)
    r.write_header(ReplayHeader());r.write_tick(ReplayTick(tick=0,sim_time=0.))
    lib=ReplayLibrary(tmp_path);lib.refresh();assert not lib.entries and not path.exists()
    r.write_result(ReplayResult(tick=1,sim_time=.2))
    assert lib.refresh() and WatchService(lib).latest_complete_path()==path
    before=path.read_bytes()
    with pytest.raises(FileExistsError):ReplayRecorder(path)
    assert path.read_bytes()==before


def test_scalar_array_and_invalid_position_records_do_not_crash(tmp_path):
    p=tmp_path/'bad.jsonl'
    p.write_text('null\n[]\n7\n'+json.dumps({'record':'tick','tick':0,'sim_time':1,'players':{'x':{'position':['bad',1]}}})+'\n')
    assert parse_replay(p)['ticks']==[] and not parse_replay(p)['complete']


def test_same_mtime_new_file_and_delete_are_discovered(tmp_path):
    a=tmp_path/'a.jsonl';make(a);lib=ReplayLibrary(tmp_path);assert lib.refresh()
    b=tmp_path/'b.jsonl';make(b);os.utime(b,ns=(a.stat().st_atime_ns,a.stat().st_mtime_ns))
    assert lib.refresh() and len(lib.entries)==2
    b.unlink();assert lib.refresh() and len(lib.entries)==1


def test_favorites_persist_and_next_queue_filters_incomplete(tmp_path):
    a=tmp_path/'a.jsonl';b=tmp_path/'b.jsonl';make(a);make(b,False)
    lib=ReplayLibrary(tmp_path);lib.refresh();lib.toggle_favorite(str(a))
    fresh=ReplayLibrary(tmp_path);fresh.refresh()
    assert str(a) in fresh.favorites and fresh.entries[0]['favorite']
    assert WatchService(fresh).next_after(str(a)) is None


def test_metadata_real_task_progress_and_selection_clicks(tmp_path):
    p=tmp_path/'a.jsonl';make(p);player=ReplayPlayer(parse_replay(p));info=player.episode_info()
    assert info['condition'].startswith('Model A') and info['controller_types']['red']=='ppo'
    assert player.current_state().players['red'].task_progress==.25
    app=SpectatorAppState();app._player_hitboxes={'red':pygame.Rect(0,0,20,20)}
    _handle_mouse_click((2,2),app,player);assert app.selected_player=='red'
    app._library_hitboxes=[(pygame.Rect(30,0,20,20),str(p))]
    assert _handle_mouse_click((32,2),app,player)==str(p)


def test_existing_demo_replay_is_complete_with_original_counts():
    data=parse_replay(Path('replays/demo_seed0007.jsonl'))
    assert data['complete'] and len(data['ticks'])==269 and len(data['events'])==153
    assert not data['neural']
