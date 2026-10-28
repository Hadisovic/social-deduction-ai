"""UI fixtures only; production monitoring always requires genuine worker records."""
import json
import os
from pathlib import Path
import tempfile
import unittest
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import pygame
from among_us_map import get_map
from spectator.v2.application import Application
from spectator.v2.preferences import Preferences
from spectator.v2.data import ReplayIndex


def fixture(seed):
    return dict(header=dict(schema='spectator-v1.0',scenario_seed=seed,config=dict(update=0,environment=0)),
        ticks=[dict(tick=1,sim_time=1.,phase='roaming',players={'p0':dict(position=[0,0],velocity=[0,0],status='alive',role='crewmate',color='Red',intention='wait')},bodies={})],
        events=[],neural=[],rewards=[],result=None,complete=False,updated_unix=1,live_snapshot=True)


class SpectatorMonitorTests(unittest.TestCase):
    def test_brief_snapshot_lock_is_not_fatal_but_persistent_failure_is(self):
        from patrick_jane.live import LiveBridge
        from types import SimpleNamespace
        from unittest.mock import patch
        import time
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'local-runs') as directory:
            bridge=LiveBridge(SimpleNamespace(game=SimpleNamespace(is_terminal=False)),directory,{'environment':0})
            bridge.recorder=SimpleNamespace(recorder=SimpleNamespace(data=lambda:{'fixture':True}))
            with patch('patrick_jane.progress.atomic_json',side_effect=PermissionError()):bridge.publish(force=True)
            self.assertIsNone(bridge.error);self.assertEqual(bridge.transient_write_failures,1)
            with patch('patrick_jane.progress.atomic_json'):bridge.publish(force=True)
            self.assertIsNone(bridge.first_write_failure)
            bridge.first_write_failure=time.monotonic()-11
            with patch('patrick_jane.progress.atomic_json',side_effect=PermissionError()):bridge.publish(force=True)
            self.assertIn('Persistent',bridge.error)

    def test_publisher_retries_a_brief_windows_reader_lock(self):
        from spectator.v2.shared_io import open_shared,read_json
        from patrick_jane.progress import atomic_json
        import threading,time
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'local-runs') as directory:
            root=Path(directory);path=root/'snapshot.json';errors=[]
            path.write_text('{"version":1}')
            def publish():
                try:atomic_json(path,dict(version=2),attempts=20)
                except BaseException as exc:errors.append(exc)
            with open_shared(path) as previous:
                writer=threading.Thread(target=publish);writer.start();time.sleep(.015)
                self.assertEqual(json.load(previous),{'version':1})
            writer.join(1.)
            self.assertFalse(errors);self.assertFalse(writer.is_alive())
            self.assertEqual(read_json(path),{'version':2})

    def test_live_switch_never_displays_stale_scenario_as_selected_instance(self):
        pygame.init()
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'local-runs') as directory:
            root=Path(directory);path=root/'env-00.json';path.write_text(json.dumps(fixture(10)))
            app=Application(Preferences(root/'prefs.json'),get_map())
            try:
                app.state.status=dict(schema='patrick-jane-progress-v1',monitor_dir=str(root),
                    environment_progress=dict(update=0,streams=[dict(environment=0,seed=11,snapshot=str(path))]))
                app.state.index=ReplayIndex(fixture(9));app.dispatch(('live_environment',0));app.poll()
                self.assertIsNone(app.state.index)
                path.write_text(json.dumps(fixture(11)));app.last_live_poll=0;app.poll()
                self.assertEqual(app.state.index.header['scenario_seed'],11)
                self.assertEqual(app.state.index.frame().sim_time,1.)
                app.set_pref('mode','follow_latest');self.assertIsNone(app.state.live_environment)
            finally:app.close();pygame.quit()

    def test_sabotage_deadline_and_resolution_are_projected_from_records(self):
        data=fixture(1);data['events']=[dict(kind='sabotage_started',time=0.,sabotage='oxygen',deadline=30.)]
        index=ReplayIndex(data);index.player.seek_to_time(1.)
        self.assertEqual(index.sabotage(),dict(kind='oxygen',remaining=29.))
        data['events'].append(dict(kind='sabotage_cancelled',time=1.,sabotage='oxygen'))
        index=ReplayIndex(data);index.player.seek_to_time(1.);self.assertIsNone(index.sabotage())


if __name__=='__main__':unittest.main()
