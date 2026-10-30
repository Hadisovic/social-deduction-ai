"""Lobby settings must reach an inference game without changing historical runs."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def api():
    return importlib.import_module('lizbon_viewer.lobby')


def model(root):
    m = api().ModelChoice('test', 'Test weights', root/'weights.pt', '--transfer-v2')
    m.path.write_bytes(b'weights')
    return m


def test_custom_counts_become_real_host_profile():
    m = api()
    p = m.make_profile(m.Draft(players=15, impostors=3, common=2, long=3, short=5), 'ui-test')
    assert p.settings.lobby.players == 15 and p.settings.lobby.impostors == 3
    assert p.effective_counts == (2, 3, 5)
    assert p.settings.lobby.kill_cooldown_seconds == 45


def test_invalid_combination_and_bad_types_rejected():
    m = api()
    with pytest.raises(ValueError, match='at least 9'):
        m.make_profile(m.Draft(players=4, impostors=3), 'bad')
    with pytest.raises(ValueError):
        m.make_profile(m.Draft(players=True), 'bad')
    with pytest.raises(ValueError):
        m.make_profile(m.Draft(short=6), 'bad')


def test_zero_counts_use_real_engine_fallback():
    m = api()
    assert m.make_profile(m.Draft(common=0,long=0,short=0), 'zero').effective_counts == (0,0,1)


def test_save_is_exclusive_and_draft_reopens(tmp_path):
    m = api(); c = m.LobbyController(ROOT, tmp_path, storage=tmp_path/'ui')
    c.draft.players=15; c.draft.impostors=3; c.draft.short=5
    a=c.save(); b=c.save()
    assert a != b and a.exists() and b.exists()
    assert json.loads(a.read_text())['settings']['tasks']['short'] == 5
    restored=m.LobbyController(ROOT,tmp_path,storage=tmp_path/'ui')
    assert restored.draft.players==15 and restored.draft.short==5


def test_corrupt_saved_draft_does_not_crash(tmp_path):
    m=api(); tmp_path.joinpath('draft.json').write_text('{broken')
    c=m.LobbyController(ROOT,tmp_path,storage=tmp_path)
    assert c.draft.players==12 and 'saved' in c.notice.lower()


def test_collecting_viewer_never_opens_critical_producer_status(tmp_path,monkeypatch):
    m=api(); process=SimpleNamespace(pid=123,poll=lambda:None)
    c=m.LobbyController(ROOT,tmp_path,storage=tmp_path/'ui',spawn=lambda *a,**k:process)
    c.models=[model(tmp_path)];c.draft.model='test';c.start()
    def forbidden(path):raise AssertionError('A live observer must not lock producer status')
    monkeypatch.setattr(m,'read_json',forbidden)
    for _ in range(5):c.last_read=0;c.poll()


def test_start_uses_smoke_partition_weights_and_saved_counts(tmp_path):
    m=api(); calls=[]
    process=SimpleNamespace(pid=123, poll=lambda:None)
    def spawn(command,**kwargs):
        calls.append((command,kwargs)); return process
    c=m.LobbyController(ROOT,tmp_path,storage=tmp_path/'ui',spawn=spawn)
    c.models=[model(tmp_path)]; c.draft.model='test'
    c.draft.players=15; c.draft.impostors=3; c.draft.short=5
    c.start()
    command=calls[0][0]
    assert command[command.index('--mode')+1]=='smoke'
    assert '--resume-optimizer' not in command and '--transfer-v2' in command
    assert command[command.index('--episodes')+1]=='1'
    assert not Path(command[command.index('--output')+1]).exists()
    p=json.loads(Path(command[command.index('--profile')+1]).read_text())
    assert p['settings']['lobby']['players']==15 and p['settings']['tasks']['short']==5
    with pytest.raises(ValueError,match='already running'):c.start()
    assert len(calls)==1


def test_missing_model_prevents_launch_and_error_retains_logs(tmp_path):
    m=api(); c=m.LobbyController(ROOT,tmp_path,storage=tmp_path/'ui')
    c.models=[]
    with pytest.raises(ValueError,match='weights'):c.start()
    assert not list((tmp_path/'ui').glob('game-*'))
    c.models=[model(tmp_path)]; c.draft.model='test'
    def fail(*args,**kwargs):raise OSError('cannot launch')
    c.spawn=fail
    with pytest.raises(OSError,match='cannot launch'):c.start()
    assert c.run_dir.joinpath('profile.json').exists() and not c.running


def test_finished_failure_and_cancellation_are_not_wins(tmp_path,monkeypatch):
    m=api(); code=[None]
    process=SimpleNamespace(pid=123,poll=lambda:code[0])
    c=m.LobbyController(ROOT,tmp_path,storage=tmp_path/'ui',spawn=lambda *a,**k:process)
    c.models=[model(tmp_path)]; c.draft.model='test'; c.start()
    c.output.mkdir(); c.output.joinpath('failure.json').write_text(json.dumps({'message':'bad geometry'}))
    code[0]=1;c.poll()
    assert 'bad geometry' in c.notice and not c.running
    code[0]=None;c.start()
    stopped=[];monkeypatch.setattr(m,'terminate_owned',lambda p:stopped.append(p))
    c.stop()
    assert stopped==[process] and 'cancelled' in c.notice.lower()
    assert c.run_dir.joinpath('viewer-stop.json').exists()


def test_time_limit_terminates_only_owned_process(tmp_path,monkeypatch):
    m=api(); process=SimpleNamespace(pid=123,poll=lambda:None)
    c=m.LobbyController(ROOT,tmp_path,storage=tmp_path/'ui',spawn=lambda *a,**k:process)
    c.models=[model(tmp_path)];c.draft.model='test';c.start();c.started-=10000
    stopped=[];monkeypatch.setattr(m,'terminate_owned',lambda p:stopped.append(p))
    c.poll()
    assert stopped==[process] and 'limit' in c.notice.lower()


def test_no_replay_launcher_opens_lobby():
    r=subprocess.run([sys.executable,str(ROOT/'run_lizbon.py'),'--dry-run'],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    data=json.loads(r.stdout)
    assert '--lobby' in data['command']
    assert '--mode' not in data['command'] or 'train' not in data['command']


def test_native_dashboard_real_controls_and_small_window(tmp_path):
    source=os.environ.get('JANE_UPSTREAM_REVIEW_ROOT')
    if not source:
        local=ROOT/'local-runs/jane-impostor-v1/source.json'
        if local.exists():source=json.loads(local.read_text())['source_root']
    if not source:pytest.skip('Pinned native source needed for Pygame integration')
    script=r'''
import json,sys,time
from pathlib import Path
import pygame
from jane.impostor.v1.bootstrap import activate
activate(Path(sys.argv[1]))
from lizbon_viewer.native import install
install()
from lizbon_viewer.lobby_ui import install as install_lobby
install_lobby(Path(sys.argv[1]),storage=Path(sys.argv[2]))
from spectator.v2.application import Application
from spectator.v2.preferences import Preferences
from among_us_map import get_map
pygame.init()
app=Application(Preferences(Path(sys.argv[2])/'prefs.json'),get_map())
app.screen=pygame.display.set_mode((1500,900))
def draw():app.draw();return {c.key:c for c in app.ui.controls}
def click(key):
    controls=draw();c=controls[key];assert c.enabled,key
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=c.rect.center))
click('page-Lobby')
assert app.state.scene=='Lobby'
for _ in range(3):click('lobby-players-plus')
click('lobby-impostors-plus');click('lobby-short-plus')
assert app.state.lobby.draft.players==15 and app.state.lobby.draft.impostors==3
assert app.state.lobby.draft.short==3
click('lobby-save')
p=json.loads(app.state.lobby.saved.read_text())
assert p['settings']['lobby']['players']==15
assert p['settings']['tasks']['short']==3
app.state.lobby.draft.players=4
assert not draw()['lobby-start'].enabled
app.state.lobby.draft.impostors=1
assert draw()['lobby-save'].enabled
app.screen=pygame.display.set_mode((1000,700))
app.state.scrolls['lobby']=400
controls=draw()
assert controls['lobby-save'].rect.colliderect(app.renderer.layout['body'])
app.screen=pygame.display.set_mode((1500,900));app.state.scrolls['lobby']=0
app.state.lobby.draft.players=15;app.state.lobby.draft.impostors=3;draw()
pygame.image.save(app.screen,str(Path(sys.argv[2])/'lobby.png'))
# Cancelled and failed games retain a labelled, bounded partial replay.
c=app.state.lobby;c.output=Path(sys.argv[2])/'partial';c.run_dir=c.output
(c.output/'live').mkdir(parents=True)
snapshot=dict(header=dict(agent_roles={'p0':'impostor'},config={}),
    ticks=[dict(tick=1,sim_time=.2,phase='roaming',players={'p0':dict(
        color='red',role='impostor',status='alive',position=[0,0],velocity=[0,0],
        intention='wait',interaction_task=None,cooldown=0,meetings_remaining=1)})],
    events=[],neural=[],rewards=[],result=None,updated_unix=1)
(c.output/'live/env-00.json').write_text(json.dumps(snapshot))
for cancelled in (True,False):
    c.finished=True;c.stop_requested=cancelled;c.notice='cancelled' if cancelled else 'failed'
    app.dispatch(('lobby_watch',))
    assert app.state.index is not None and app.state.live_environment is None
    assert app.state.lobby_partial
# A failed game with no snapshot must not relabel another game's old index.
partial_output=c.output;c.output=Path(sys.argv[2])/'empty-failure';c.output.mkdir()
app.dispatch(('lobby_watch',))
assert app.state.index is None and app.state.current_path is None
c.output=partial_output
# Attaching live invalidates an older asynchronous library request/result.
c.finished=False;c.stop_requested=False
app.state.library.requested='old.jsonl';app.state.library.load_result=('old.jsonl',{},None)
app.dispatch(('lobby_watch',))
assert app.state.library.requested is None and app.state.library.load_result is None
assert app.state.library.directory.resolve()==c.storage.resolve()
c.finished=True
# Completed replay loading must replace the live index, not merely clear the
# live_environment flag while the old bounded snapshot is still displayed.
from spectator.replay_format import SCHEMA_VERSION
header={**snapshot['header'],'record':'header','schema':SCHEMA_VERSION}
tick={**snapshot['ticks'][0],'record':'tick','tick':0,'sim_time':0}
result=dict(record='result',tick=0,sim_time=0,winner='crewmate',reason='TASKS_COMPLETED')
replay=c.output/'match-001.jsonl'
replay.write_text(''.join(json.dumps(r)+'\n' for r in (header,tick,result)))
(c.output/'results.json').write_text('{}');c.stop_requested=False
app.dispatch(('lobby_watch',));assert app.state.index is None
deadline=time.monotonic()+5
while app.state.index is None and time.monotonic()<deadline:app.poll();time.sleep(.02)
assert app.state.current_path==str(replay.resolve())
assert not app.state.index.data.get('live_snapshot',False)
assert not app.state.playing and app.state.index.player.current_sim_time==0
app.close();pygame.quit()
print('Native lobby clicks/save/validation/scroll passed')
'''
    env={**os.environ,'SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy'}
    result=subprocess.run([sys.executable,'-c',script,str(source),str(tmp_path)],
        cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert (tmp_path/'lobby.png').is_file()
