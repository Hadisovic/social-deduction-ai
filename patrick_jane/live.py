"""Bounded observer-only snapshots using the established replay projections.

No actor receives these records. Rendering and file polling never gate physics.
Only one selected instance is instrumented; complete recordings stay separate.
"""
from collections import deque,defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic,time
from spectator.recorder import MatchRecorder
from spectator.replay_format import ReplayHeader,SCHEMA_VERSION


class Ring:
    def __init__(self,header):
        self.header=header;self.ticks=deque(maxlen=32);self.events=deque(maxlen=2048)
        self.neural={};self.rewards={};self.components=defaultdict(lambda:defaultdict(float));self.result=None

    def write_tick(self,value):self.ticks.append(asdict(value))
    def write_event(self,value):self.events.append(dict(value))
    def write_neural(self,value):self.neural[value.player_id]=asdict(value)
    def write_reward(self,value):
        record=asdict(value)
        for k,v in record['components'].items():self.components[value.player_id][k]+=v
        record['components']=dict(self.components[value.player_id]);self.rewards[value.player_id]=record
    def write_result(self,value):self.result=asdict(value)

    def data(self):
        return dict(header=self.header,ticks=list(self.ticks),events=list(self.events),
            neural=list(self.neural.values()),rewards=list(self.rewards.values()),result=self.result,
            complete=self.result is not None,tags=[],live_snapshot=True,bounded_history=True,updated_unix=time())


class LiveRecorder(MatchRecorder):
    def __init__(self,match,metadata):
        self.match=match;self._last_event_count=0
        game=match.game
        header=ReplayHeader(schema=SCHEMA_VERSION,game_schema='patrick-jane-v1',
            source_version=metadata.get('source','working-tree'),training_session='patrick-jane-live',
            scenario_seed=game.seed,agent_roles={pid:p.role.value for pid,p in game.players.items()},
            controller_types={pid:'patrick-jane-ppo' if pid in match.crew else 'scripted-impostor' for pid in game.players},
            checkpoint_path=metadata.get('checkpoint','fresh-initialization-seed17'),
            model_architecture='Patrick Jane: shared GRU256, five heads',config={**self._safe_config(game),**metadata,'patrick_jane_schema':'patrick-jane-v1'})
        self.recorder=Ring(asdict(header));self.after_step()


class LiveBridge:
    def __init__(self,match,directory,metadata):
        self.match=match;self.directory=Path(directory);self.metadata=metadata
        self.environment=metadata['environment'];self.recorder=None;self.polled=-1e20;self.published=-1e20
        self.seconds=0.;self.error=None
        self.transient_write_failures=0;self.first_write_failure=None

    def check(self):
        now=monotonic()
        if now-self.polled<1:return
        self.polled=now;focus=0
        try:focus=int(json.loads((self.directory/'focus.json').read_text())['environment'])
        except (OSError,ValueError,KeyError,TypeError):pass
        if focus==self.environment and self.recorder is None:self.recorder=LiveRecorder(self.match,self.metadata)
        elif focus!=self.environment:
            self.recorder=None;self.first_write_failure=None

    @property
    def active(self):return self.recorder is not None

    def neural(self,pid,record,packet,nav):
        if not self.recorder:return
        from .replay import neural_record
        neural_record(self.recorder,self.match,pid,record,packet,nav)

    def reward(self,when,pid,components,cumulative):
        if not self.recorder:return
        from spectator.replay_format import ReplayReward
        self.recorder.recorder.write_reward(ReplayReward(tick=self.match.game.tick,sim_time=when,
            player_id=pid,reward=sum(components.values()),cumulative_reward=cumulative,components=components))

    def step(self):
        if not self.recorder:return
        started=monotonic();self.recorder.after_step()
        game=self.match.game
        self.recorder.recorder.write_event(dict(tick=game.tick,time=game.time,kind='live_sabotage',
            channel='spectator',sabotage=game.sabotage,deadline=game.sabotage_deadline))
        self.publish(force=game.is_terminal)
        self.seconds+=monotonic()-started

    def publish(self,force=False):
        if not self.recorder:return
        started=monotonic();game=self.match.game
        if force or started-self.published>=.5:
            if game.is_terminal:self.recorder.finish()
            try:
                self.directory.mkdir(parents=True,exist_ok=True)
                from .progress import atomic_json
                path=self.directory/f'env-{self.environment:02d}.json'
                atomic_json(path,self.recorder.recorder.data(),attempts=8 if force else 1,compact=True)
                self.published=started;self.first_write_failure=None
            except PermissionError:
                self.transient_write_failures+=1
                if self.first_write_failure is None:self.first_write_failure=started
                if started-self.first_write_failure>=10.:
                    self.error='Persistent snapshot permission failure (10 seconds)'
            except OSError as exc:self.error=type(exc).__name__ # Observer cannot stop simulation.
