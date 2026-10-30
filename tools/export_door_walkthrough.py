"""Export actual native viewer frames from a recorded game; never simulates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('SDL_VIDEODRIVER','dummy')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
    from jane.impostor.v1.bootstrap import activate
    activate(args.source_root)
    from lizbon_viewer.native import install
    install()
    import pygame
    from PIL import Image
    from among_us_map import get_map
    from spectator.replay_format import parse_replay
    from spectator.v2.application import Application
    from spectator.v2.preferences import Preferences
    from spectator.v2.data import ReplayIndex
    before=hashlib.sha256(args.replay.read_bytes()).hexdigest()
    data=parse_replay(args.replay)
    if not data['complete']:raise ValueError('Use a genuine completed replay')
    pygame.init();prefs=Preferences(ROOT/'local-runs/door-viewer-check/export-preferences.json')
    for name,value in dict(omniscient=True,reveal_roles=True,roster=False,inspector=False,tasks=False).items():prefs.set(name,value)
    app=Application(prefs,get_map());app.screen=pygame.display.set_mode((1500,900))
    app.state.index=ReplayIndex(data);app.state.selected='p1';app.state.current_path=str(args.replay)
    app.renderer.map.zoom=2.3;app.renderer.map.center=(-14.,-.8)
    # Derive the demo window from physical telemetry, not a guessed expiry.
    from lizbon_viewer.doors import DoorTimeline
    tracks=DoorTimeline(app.state.index.a2_states).tracks
    edges=tracks['11:11'].edges
    close=next(e.time for e in edges if e.target==1)
    reopen=next(e.time for e in edges if e.time>close and e.target==0)
    times=[close-.15+.025*i for i in range(25)]+[close+2.2,close+6.2,reopen-.1]+[reopen-.05+.025*i for i in range(25)]
    frames=[]
    try:
        for when in times:
            app.state.index.player.seek_to_time(when);app.draw()
            frames.append(Image.frombytes('RGB',app.screen.get_size(),pygame.image.tobytes(app.screen,'RGB')))
        for label,when in [('open',close-.15),('closing',close+.125),('closed',close+2.2),('opening',reopen+.125)]:
            app.state.index.player.seek_to_time(when);app.draw();pygame.image.save(app.screen,str(args.output/(label+'.png')))
        durations=[50]*25+[350,350,350]+[50]*25
        # A shared palette preserves delta compression; native PNGs stay full size.
        previews=[f.resize((1000,600),Image.Resampling.LANCZOS) for f in frames]
        palette=previews[0].quantize(colors=128)
        compact=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in previews]
        compact[0].save(args.output/'recorded_door_animation.gif',save_all=True,append_images=compact[1:],
                       duration=durations,loop=0,optimize=True)
    finally:app.close();pygame.quit()
    after=hashlib.sha256(args.replay.read_bytes()).hexdigest()
    if after!=before:raise ValueError('Replay unexpectedly changed')
    proof=dict(source_replay=args.replay.name,replay_sha256=before,unchanged=True,
        source_seed=data['header']['scenario_seed'],controller_context='Initial-policy baseline, not an updated-model evaluation',
        renderer='Actual native Pygame Application with the presentation-only door adapter',
        frame_times=times,playback='Transitions shown at half speed; quiet middle sampled rather than simulated',
        first_recorded_closed_seconds=close,recorded_open_seconds=reopen,
        model_inference_calls=0,training_steps=0)
    (args.output/'walkthrough.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(gif=str(args.output/'recorded_door_animation.gif'),frames=len(frames),replay_unchanged=True)))


if __name__=='__main__':main()
