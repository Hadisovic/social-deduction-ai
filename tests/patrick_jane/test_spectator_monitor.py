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
