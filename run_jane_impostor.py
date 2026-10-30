"""Launch the existing independent A2 spectator; this command never trains."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--replay',type=Path)
    parser.add_argument('--watch',type=Path)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--selected')
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--seek',type=float,default=0.)
    parser.add_argument('--paused',action='store_true')
    parser.add_argument('--preferences',type=Path,default=ROOT/'local-runs/jane-impostor-v1/viewer-preferences.json')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.source_root is None:
        path=ROOT/'local-runs/jane-impostor-v1/source.json'
        if not path.exists():parser.error('Provide --source-root with the pinned Patrick Jane checkout')
        args.source_root=Path(json.loads(path.read_text())['source_root'])
    source=args.source_root.resolve()
    if not (source/'run_spectator_v2.py').is_file():parser.error('Source root lacks the native spectator')
    if not args.replay and not args.watch:parser.error('Choose --replay or --watch; no game is generated automatically')
    runtime=ROOT/'local-runs/jane-impostor-v1/runtime/Scripts'
    python=runtime/'pythonw.exe'
    if not python.is_file():python=runtime/'python.exe'
    if not python.is_file():python=Path(sys.executable)
    command=[str(python),str(ROOT/'run_lizbon_viewer.py'),'--source-root',str(source),
             '--omniscient','--width','1500','--height','900']
    command+=['--preferences',str(args.preferences.resolve()),'--speed',str(args.speed)]
    if args.replay:command+=['--replay',str(args.replay.resolve()),'--mode','manual','--seek',str(args.seek)]
    if args.watch:command+=['--watch',str(args.watch.resolve()),'--status-file',str(args.watch.resolve()/'status.json'),
                             '--live-environment','0','--mode','follow_latest']
    if args.selected is None and args.replay:
        with args.replay.open(encoding='utf-8') as handle:header=json.loads(next(handle))
        args.selected=next((pid for pid,role in header.get('agent_roles',{}).items() if role=='impostor'),None)
    if args.selected:command+=['--selected',args.selected]
    if args.paused:command+=['--paused']
    if args.capture:command+=['--capture',str(args.capture.resolve())]
    if args.max_frames:command+=['--max-frames',str(args.max_frames)]
    if args.dry_run:
        print(json.dumps(dict(command=command,mode='playback' if args.replay else 'live observer; no training'),indent=2));return
    folder=args.watch or args.replay.parent
    log=folder/'viewer.log'
    with log.open('a',encoding='utf-8') as handle:
        process=subprocess.Popen(command,cwd=source,stdout=handle,stderr=subprocess.STDOUT)
    print(json.dumps(dict(viewer_pid=process.pid,mode='playback' if args.replay else 'live observer',log=str(log))))


if __name__=='__main__':main()
