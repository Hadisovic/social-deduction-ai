"""Render specific behavioral moments in the approved viewer; not general UI QA."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spectator-source',required=True,type=Path)
    p.add_argument('--historical',required=True,type=Path);p.add_argument('--new-replay',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.append(str(args.spectator_source.resolve()));os.environ['SDL_VIDEODRIVER']='dummy'
    import pygame
    from among_us_map import get_map
    from spectator.v2.application import Application
    from spectator.v2.preferences import Preferences
    from spectator.v2.data import ReplayIndex
    from spectator.replay_format import parse_replay
    pygame.init();app=Application(Preferences(args.output/'viewer-preferences.json'),get_map())
    app.screen=pygame.display.set_mode((1800,1050));app.state.prefs.set('omniscient',True)
    app.state.prefs.set('reveal_roles',True);app.state.prefs.set('reveal_ballots',True)
    inspected=[]
    try:
        for path,actor,when,tab,name in ((args.historical,'p4',126.2,'Mind','historical-defense'),
                (args.historical,'p4',129.6,'Mind','historical-vote'),
                (args.historical,'p4',129.6,'Belief','historical-belief'),
                (args.new_replay,None,61.,'Mind','mixture-engineering')):
            index=ReplayIndex(parse_replay(path));index.player.seek_to_time(when)
            if actor is None:actor=next(pid for pid in index.frame().players if index.header['controller_types'][pid]=='ppo')
            app.state.index=index;app.state.current_path=str(path.resolve());app.state.selected=actor;app.state.tab=tab
            app.draw();pygame.image.save(app.screen,str(args.output/(name+'.png')))
            neural=index.neural(actor)
            inspected.append(dict(view=name,actor=actor,time=when,action=neural.get('action_kind') if neural else None,
                belief=neural.get('suspicion_probabilities') if neural else None,private_observation_available=index.evidence(actor)['private'] is not None))
        assert inspected[0]['action']=='claim_defend' and inspected[1]['action']=='vote'
        with (args.output/'private-inspection.json').open('x') as stream:json.dump(inspected,stream,indent=2)
        print('Four behavioral views rendered from real recordings in unchanged Spectator V2.')
    finally:app.close();pygame.quit()


if __name__=='__main__':main()
