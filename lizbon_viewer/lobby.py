"""Local lobby drafts and owned inference games; no neural or engine mutation."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4


@dataclass
class Draft:
    players: int = 12
    impostors: int = 2
    common: int = 1
    long: int = 1
    short: int = 2
    model: str = 'classic-development'
    minutes: int = 10
    greedy: bool = False


@dataclass(frozen=True)
class ModelChoice:
    key: str
    label: str
    path: Path
    argument: str


ROOT = Path(__file__).resolve().parents[1]
LIMITS = dict(players=(4,15), impostors=(1,3), common=(0,2), long=(0,3), short=(0,5))
BUDGETS = (2,5,10,20,30)


def make_profile(draft, name, root=ROOT):
    from jane.classic_v2.settings import HostProfile, load_profile
    base = load_profile(root/'jane/configs/classic-v2/default_12p.json').settings
    return HostProfile(replace(base, profile_id=name,
        lobby=replace(base.lobby, players=draft.players, impostors=draft.impostors),
        tasks=replace(base.tasks, common=draft.common, long=draft.long, short=draft.short)))


def read_json(path):
    try:
        # Release the file before decoding. Only observer snapshots are read
        # during play; critical producer status/results wait until process exit.
        data=json.loads(path.read_bytes())
        return data if isinstance(data,dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def terminate_owned(process):
    """Only this Popen handle's still-running process tree is eligible."""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,check=False)
    else:
        os.killpg(process.pid,15)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


class LobbyController:
    def __init__(self, root, source, *, storage=None, spawn=None):
        self.root=Path(root); self.source=Path(source).resolve()
        self.storage=Path(storage) if storage else self.root/'local-runs/lizbon-dashboard'
        self.spawn=spawn or subprocess.Popen
        choices=(
            ModelChoice('classic-development','Classic V2 development · 24 PPO passes',
                self.root/'local-runs/lizbon-classic-v2/fix-20261031/checkpoint-003-migrated.pt','--transfer-v2'),
            ModelChoice('lizbon-v1','Preserved Lizbon V1 · 8 PPO passes',
                self.root/'artifacts/lizbon-v1/pilot-20261030/checkpoint-001.pt','--warm-start-v1'))
        self.models=[m for m in choices if m.path.is_file()]
        self.draft=Draft(); self.notice='Choose a lobby, then start a new AI game.'
        self.saved=None; self.process=None; self.run_dir=None; self.output=None
        self.started=0.; self.active_minutes=0; self.finished=False
        self.last_read=0.; self.progress={}; self.stop_requested=False
        path=self.storage/'draft.json'
        if path.exists():
            try:
                self.draft=Draft(**json.loads(path.read_text(encoding='utf-8')))
                self.validate()
            except (OSError,ValueError,TypeError):
                self.draft=Draft(); self.notice='Saved lobby was unreadable; defaults restored.'
        if self.models and not any(m.key==self.draft.model for m in self.models):
            self.draft.model=self.models[0].key

    @property
    def running(self):
        return self.process is not None and not self.finished and not self.stop_requested and self.process.poll() is None

    def validate(self):
        if type(self.draft.greedy) is not bool or type(self.draft.minutes) is not int or self.draft.minutes not in BUDGETS:
            raise ValueError('Choose a supported wall-time limit and playback policy.')
        if type(self.draft.model) is not str:
            raise ValueError('Choose model weights.')
        return make_profile(self.draft,'dashboard-preview',self.root)

    def error(self):
        try: self.validate()
        except ValueError as exc: return str(exc)
        return ''

    def persist(self):
        self.validate()
        write_json(self.storage/'draft.json',asdict(self.draft))

    def save(self):
        from jane.classic_v2.settings import save_profile
        self.validate(); name='lobby-'+uuid4().hex[:12]
        path=self.storage/'profiles'/(name+'.json')
        save_profile(path,make_profile(self.draft,name,self.root))
        self.persist(); self.saved=path
        self.notice='Lobby saved. Existing games keep their original settings.'
        return path

    def start(self):
        if self.running:
            raise ValueError('An AI game is already running. Watch or stop it first.')
        self.validate()
        model=next((m for m in self.models if m.key==self.draft.model and m.path.is_file()),None)
        if model is None:
            raise ValueError('No available Lizbon weights. Restore a listed checkpoint first.')
        self.persist()
        name='game-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8]
        self.run_dir=self.storage/name; self.run_dir.mkdir(parents=True,exist_ok=False)
        from jane.classic_v2.settings import save_profile
        profile_path=self.run_dir/'profile.json'
        save_profile(profile_path,make_profile(self.draft,name,self.root))
        self.output=self.run_dir/'match'; self.saved=profile_path
        runtime=self.root/'local-runs/jane-impostor-v1/runtime/Scripts/python.exe'
        python=runtime if runtime.is_file() else Path(sys.executable)
        offset=uuid4().int % 1000
        command=[str(python),str(self.root/'run_lizbon_classic.py'),
            '--source-root',str(self.source),'--profile',str(profile_path),
            '--output',str(self.output),'--mode','smoke','--episodes','1',
            '--scenario-offset',str(offset),'--smoke-observations','1000000',
            '--max-minutes',str(self.draft.minutes+1),'--match-wall-seconds',str(self.draft.minutes*60),
            model.argument,str(model.path)]
        if self.draft.greedy: command.append('--deterministic')
        write_json(self.run_dir/'viewer-launch.json',dict(
            purpose='interactive inference only; no training/selection/final tests',
            created_utc=datetime.now(timezone.utc).isoformat(),command=command,
            model=model.label,profile=str(profile_path),scenario_seed=9700001+offset,
            wall_limit_minutes=self.draft.minutes,new_settings_performance='unvalidated'))
        self.process=None; self.finished=False; self.stop_requested=False; self.progress={}
        self.active_minutes=self.draft.minutes; self.started=time.monotonic(); self.last_read=0.
        options=dict(cwd=self.root)
        if os.name=='nt': options['creationflags']=subprocess.CREATE_NO_WINDOW
        else: options['start_new_session']=True
        try:
            with (self.run_dir/'console.log').open('w',encoding='utf-8') as log:
                self.process=self.spawn(command,stdout=log,stderr=subprocess.STDOUT,**options)
        except OSError as exc:
            self.notice='Could not start game: '+str(exc)
            write_json(self.run_dir/'launch-error.json',dict(error=str(exc)))
            raise
        self.notice='Starting AI game. Current weights; learning is disabled.'
        return self.output

    def stop(self, reason='Game cancelled by viewer; partial artifacts retained.'):
        if not self.running: return
        terminate_owned(self.process)
        self.stop_requested=True; self.finished=True; self.notice=reason
        write_json(self.run_dir/'viewer-stop.json',dict(reason=reason,
            elapsed_wall_seconds=time.monotonic()-self.started,complete_match=False))

    def poll(self):
        if self.process is None or self.finished: return
        code=self.process.poll()
        if code is None:
            if time.monotonic()-self.started>self.active_minutes*60+30:
                self.stop('Viewer wall-time limit reached; partial game, no scored outcome.')
            elif time.monotonic()-self.last_read>.5:
                self.last_read=time.monotonic()
            return
        self.finished=True
        if code:
            failure=read_json(self.output/'failure.json')
            self.notice='Game failed: '+str(failure.get('message',failure.get('error',f'exit {code}; see console.log')))
        else:
            results=read_json(self.output/'results.json')
            matches=results.get('matches',[])
            row=matches[0] if matches else {}
            if row.get('winner') and not row.get('truncated',True):
                self.notice=f"Game complete: {row.get('winner','see replay')} · {row.get('reason','see replay')}"
            else:
                self.notice='Game ended at an engineering/time limit; no scored win or loss. Replay available.'

    def close(self):
        self.stop('Game cancelled because its dashboard closed; partial artifacts retained.')
