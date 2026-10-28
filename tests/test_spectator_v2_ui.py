"""Real input dispatch, layout bounds and read-only genuine-recording checks."""
import json
import os
from pathlib import Path
import time
import pytest

os.environ.setdefault('SDL_VIDEODRIVER','dummy')
import pygame
from among_us_map import get_map
from spectator.v2.application import Application
from spectator.v2.preferences import Preferences
from spectator.v2.data import ReplayIndex
from spectator.v2.library import Library,inspect_file
from spectator.replay_format import parse_replay
from test_spectator_v2_data import recording

REAL=Path(__file__).resolve().parents[1]/'local-runs/handoff'


@pytest.fixture
def app(tmp_path):
    pygame.init();a=Application(Preferences(tmp_path/'settings.json'),get_map())
    a.screen=pygame.display.set_mode((1800,1050));a.state.index=ReplayIndex(recording());a.state.selected='a';a.state.current_path='fixture.jsonl'
    yield a
    a.close();pygame.quit()


def click(app,key):
    app.draw();control=next(c for c in app.ui.controls if c.key==key)
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=control.rect.center));app.draw()


def test_clickable_transport_changes_actual_simulation_time(app):
    for speed in (.25,.5,1.,2.,4.,8.,16.):
        app.state.index.player.seek_to_time(0.)
        click(app,'speed-'+str(speed));assert app.state.speed==speed
        if not app.state.playing:click(app,'transport-play')
        app.advance(.1)
        assert app.state.index.player.current_sim_time==pytest.approx(min(3.,.1*speed))
        app.state.playing=False
    click(app,'transport-restart');assert app.state.index.player.current_sim_time==0
    click(app,'transport-forward');assert app.state.index.player.current_sim_time==1
    click(app,'transport-back');assert app.state.index.player.current_sim_time==0


def test_settings_debug_controls_and_persistence_are_real(app):
    click(app,'page-Settings');click(app,'settings-Debug')
    click(app,'setting-omniscient');click(app,'setting-reveal_ballots')
    app.state.index.player.seek_to_time(2.)
    assert app.state.index.meeting(app.state.prefs['omniscient'],app.state.prefs['reveal_ballots'])['ballots'][0]['target']=='i'
    loaded=Preferences(app.state.prefs.path);assert loaded['omniscient'] and loaded['reveal_ballots']
    click(app,'setting-reveal_ballots')
    assert app.state.index.meeting(True,app.state.prefs['reveal_ballots'])['ballots'][0]['state']=='hidden'


def test_all_player_slots_and_inspector_tabs_switch(app):
    for key,pid in enumerate(app.state.index.frame().players):
        app.handle(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_1+key,mod=0));assert app.state.selected==pid
    click(app,'roster-a')
    for tab in ('Belief','Rewards','Evidence','Mind'):
        click(app,'inspect-tab-'+tab);assert app.state.tab==tab


def test_exact_seek_input_and_keyboard_focus(app):
    click(app,'seek-seconds');app.state.seek_text='2.50'
    app.handle(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_RETURN,mod=0))
    assert app.state.index.player.current_sim_time==2.5
    app.handle(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_TAB,mod=0));assert app.ui.focus


def test_scrub_and_camera_do_not_mutate_recorded_positions(app):
    before=json.dumps(app.state.index.data,sort_keys=True)
    app.draw();app.dispatch(('scrub',),(app.renderer.timeline_rect.right,app.renderer.timeline_rect.centery))
    assert app.state.index.player.current_sim_time==3.
    app.renderer.map.zoom_at(1.25);app.renderer.map.pan(35,-20);app.draw()
    assert json.dumps(app.state.index.data,sort_keys=True)==before
    point=(1.,-5.);assert app.renderer.map.unproject(app.renderer.map.point(point))==pytest.approx(point,abs=.1)


@pytest.mark.parametrize('size',[(1000,700),(1280,800),(1800,1050)])
def test_controls_stay_inside_resized_window_across_pages(app,size):
    app.screen=pygame.display.set_mode(size)
    for scale in (1.,1.3):
        app.state.prefs.set('scale',scale)
        for page in ('Match','Library','Study','Settings'):
            app.state.scene=page;app.draw()
            assert all(app.screen.get_rect().contains(c.rect) for c in app.ui.controls),(size,scale,page)


def test_fullscreen_and_return_preserve_playback(app):
    app.state.index.player.seek_to_time(2.);before=app.state.index.player.current_sim_time
    app.dispatch(('fullscreen',));assert app.state.fullscreen
    app.dispatch(('fullscreen',));assert not app.state.fullscreen
    assert app.screen.get_size()==(1800,1050) and app.state.index.player.current_sim_time==before


def test_future_timeline_is_opt_in_and_truth_events_require_debug(app):
    app.state.index.player.seek_to_time(1.)
    assert all(m.sim_time<=1 for m in app.renderer.visible_markers(app.state))
    app.state.prefs.set('timeline_all',True)
    assert any(m.sim_time>1 for m in app.renderer.visible_markers(app.state))
    assert not any(getattr(m,'channel',None)=='truth' for m in app.renderer.visible_markers(app.state))


def test_missing_telemetry_pages_render_without_fake_predictions(app):
    d=recording();d['neural']=[];d['rewards']=[];app.state.index=ReplayIndex(d)
    for tab in ('Controller','Evidence','Rewards'):
        app.state.tab=tab;app.draw()
    assert app.state.index.neural('a') is None and app.state.index.reward('a') is None


def test_background_library_loads_real_file_without_unbounded_full_cache(tmp_path):
    path=REAL/'shaped_v2/ordinary-2790005.jsonl'
    if not path.is_file():pytest.skip('Private real recordings are local only')
    before=path.stat().st_mtime_ns;lib=Library();lib.set_directory(path.parent);lib.request(path)
    try:
        deadline=time.monotonic()+20;loaded=None
        while time.monotonic()<deadline:
            lib.poll();loaded=lib.take_loaded()
            if loaded:break
            time.sleep(.01)
        assert loaded and loaded[2] is None and loaded[1]['complete']
        assert len(lib.pending)<=2 and path.stat().st_mtime_ns==before
    finally:lib.close()


@pytest.mark.parametrize('label,context,seed',[('shaped_v2','ordinary',2790002),('original','body',2790000),('zero','meeting_crew',2790004)])
def test_genuine_policy_recordings_all_tabs_and_ballot_boundary(app,label,context,seed):
    path=REAL/label/(context+'-'+str(seed)+'.jsonl')
    if not path.is_file():pytest.skip('Private real recordings are local only')
    app.state.index=ReplayIndex(parse_replay(path));app.state.current_path=str(path.resolve())
    idx=app.state.index;idx.player.seek_to_time(61. if seed==2790002 else 90.4)
    app.state.selected=next(pid for pid in idx.frame().players if idx.header['controller_types'][pid]=='ppo')
    app.state.prefs.set('omniscient',True);app.state.prefs.set('reveal_roles',True)
    for tab in ('Mind','Belief','Rewards','Evidence'):
        app.state.tab=tab;app.draw()
    if label=='shaped_v2':
        hidden=idx.meeting(True,False);revealed=idx.meeting(True,True)
        assert any(b['state']=='hidden' for b in hidden['ballots'])
        assert any(b['state'] in ('vote','skip') and b['debug'] for b in revealed['ballots'])


def test_old_neural_record_without_selected_index_is_not_fabricated(app):
    d=recording();d['neural'][0]['selected_action_index']=None;app.state.index=ReplayIndex(d)
    app.draw()
    assert app.state.index.neural('a')['selected_action_index'] is None


def test_library_preserves_actual_crew_ballots_separately_from_historical_ledgers():
    summary=REAL.parents[2]/'fourcrew-reporting-2048-seed17-v1/validation.jsonl'
    if not summary.is_file():pytest.skip('Private development validation only')
    rows=[json.loads(line) for line in summary.read_text().splitlines()]
    for row in rows:
        if row['policy']!='shaped_v2' or row['context']!='ordinary' or row['seed'] not in (2790000,2790003):continue
        meta=inspect_file(REAL/row['policy']/(row['context']+'-'+str(row['seed'])+'.jsonl'))
        counts={k:sum(a['reward_counts'].get(k,0) for a in row['actors'].values()) for k in ('correct_votes','incorrect_votes','skips')}
        assert meta['correct_ballots']==counts['correct_votes']
        if row['seed']==2790003:
            # Published ballot from a subsequently ejected actor is absent in the old ledger.
            # The viewer must preserve both sources rather than silently rewrite the study.
            assert meta['wrong_ballots']==3 and counts['incorrect_votes']==2
        else:
            assert meta['wrong_ballots']==counts['incorrect_votes']
        assert meta['skips']==counts['skips']
        assert meta['neural_reports']==sum(a['reporting'].get('accepted',0) for a in row['actors'].values())


def test_all_settings_toggles_change_their_real_preferences(app):
    click(app,'page-Settings')
    categories={'Appearance':['reduced_motion','interpolate'],'Map':['labels','rooms','tasks','bodies','visibility','trails','collisions','follow','reveal_roles'],
                'Debug':['omniscient','reveal_roles','reveal_ballots','neural','rewards','evidence','unavailable'],
                'Playback':['autoplay','timeline_all'],'Interface':['roster','inspector']}
    for category,keys in categories.items():
        click(app,'settings-'+category)
        for key in keys:
            app.state.scrolls['settings']=0;app.draw()
            control=next((c for c in app.ui.controls if c.key=='setting-'+key),None)
            if control is None:
                app.state.scrolls['settings']=180;app.draw()
            old=app.state.prefs[key];click(app,'setting-'+key)
            assert app.state.prefs[key] is not old,key


def test_saved_bookmarks_are_navigable_from_the_real_menu(app):
    app.state.index.player.seek_to_time(2.);app.dispatch(('bookmark',))
    app.state.scene='Library';click(app,'bookmarks-menu')
    assert app.state.dropdown['options'][0][1][0]=='load_bookmark'
    mark=app.state.prefs['bookmarks'][-1]
    assert mark['time']==2. and mark['player']=='a'
