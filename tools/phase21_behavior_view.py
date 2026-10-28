"""Specific hybrid-decision inspection in the approved unchanged viewer."""
import argparse
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--spectator-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.append(str(a.spectator_source.resolve()));os.environ['SDL_VIDEODRIVER']='dummy'
    import pygame
    from among_us_map import get_map
    from spectator.v2.application import Application
    from spectator.v2.preferences import Preferences
    from spectator.v2.data import ReplayIndex
    from spectator.replay_format import parse_replay
    pygame.init();app=Application(Preferences(a.output/'prefs.json'),get_map());app.screen=pygame.display.set_mode((1800,1050))
    app.state.index=ReplayIndex(parse_replay(a.replay));app.state.current_path=str(a.replay.resolve());app.state.selected='p1'
    app.state.prefs.set('omniscient',True);app.state.prefs.set('reveal_roles',True);app.state.prefs.set('reveal_ballots',True)
    assert app.state.index.header['config']['features']['witness_priority']
    try:
        for name,when in (('hybrid-report',45.2),('hybrid-meeting',60.4)):
            app.state.index.player.seek_to_time(when);app.draw();pygame.image.save(app.screen,str(a.output/(name+'.png')))
        print('Two actual hybrid behavioral views rendered. Probabilities are post-intervention; raw NN values remain in diagnostic logs.')
    finally:app.close();pygame.quit()


if __name__=='__main__':main()
