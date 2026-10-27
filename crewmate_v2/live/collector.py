"""Observer-only replay adapter; reuse the frozen collection and PPO kernels."""
import hashlib
import multiprocessing as mp
from pathlib import Path
from time import perf_counter
import torch
from crewmate_v2.config import LobbySettings
from crewmate_v2.collection.bank import EnvironmentBank
from crewmate_v2.collection.collector import Collector
from crewmate_v2.collection.config import context
from .protocol import spectator_import

LABELS={'original':'Model A — Original Rewards','shaped':'Model B — Shaped Rewards'}


class RecordingBank(EnvironmentBank):
    def __init__(self,*args,recording,**kwargs):
        self.recording=recording;self.recorders={};self.statistics={};self.ready=False
        self.arm=kwargs['arm']
        self.recorder_module=spectator_import(recording['spectator_source'])
        super().__init__(*args,**kwargs)
        self.ready=True
        for slot in self.envs:self.start(slot)

    def reset(self,slot):
        super().reset(slot)
        if self.ready:self.start(slot)

    def start(self,slot):
        if slot not in self.recording['streams']:return
        env=self.envs[slot];match=env.match
        selected=context(self.partition,slot,self.episodes[slot],self.streams)
        folder=Path(self.recording['replay_dir'])/self.arm
        path=folder/f'stream-{slot}-episode-{self.episodes[slot]:05d}-seed-{selected["seed"]}.jsonl'
        types={pid:'ppo' if pid==env.focal else 'scripted' for pid in match.game.players}
        recorder=self.recorder_module.MatchRecorder(match,path,episode=self.episodes[slot],
            training_session=self.recording['session'],model_architecture='V2Policy 51778 parameters',
            source_version=self.recording['source_version'],controller_types=types,
            metadata=dict(training_condition=LABELS[self.arm],stream=slot,worker=slot%self.recording['workers'],
                          focal_player=env.focal,recording='real PPO training environment; completed matches only',
                          reward_semantics='Raw per-option components; PPO uses physical-time discounted return'))
        recorder.after_step()  # Actual initial simulator state.
        self.recorders[slot]=recorder
        self.statistics.setdefault(slot,dict(completed=[],capture_seconds=0.,frames=0))
        original_tick=match.tick
        def observed_tick(intent=None):
            result=original_tick(intent)
            started=perf_counter();recorder.after_step()
            self.statistics[slot]['capture_seconds']+=perf_counter()-started
            self.statistics[slot]['frames']+=1
            if recorder.recorder.error:raise OSError('Replay capture failed: '+recorder.recorder.error)
            return result
        match.tick=observed_tick

    def step(self,payload):
        actions={slot:(item['action'] if isinstance(item,dict) else item) for slot,item in payload.items()}
        active={slot:self.recorders[slot] for slot in actions if slot in self.recorders}
        for slot,recorder in active.items():
            item=payload[slot]
            if isinstance(item,dict):
                options=self.envs[slot].options
                labels=[f'{i:02d}: {c.kind} '+str(c.target_player or c.task_id or '') for i,c in enumerate(options)]
                probs={label:float(item['probabilities'][i]) for i,label in enumerate(labels)}
                recorder.record_neural(self.envs[slot].focal,labels[actions[slot]],
                    probabilities=probs,value_estimate=item['value'],entropy=item['entropy'],legal_actions=labels,
                    masked_actions=list(range(len(options),96)),training_step=item['training_step'],
                    policy_update=item['policy_update'],selected_action_index=actions[slot])
        rows=super().step(actions)
        from spectator.replay_format import ReplayReward
        for slot,recorder in active.items():
            row=rows[slot];game=recorder.match.game
            recorder.recorder.write_reward(ReplayReward(tick=game.tick,sim_time=game.time,player_id=recorder.match.focal,
                reward=row['info']['undiscounted_reward'],cumulative_reward=sum(recorder.match.ledger.components.values()),
                components=row['info']['reward_components']))
            if row['done']:
                recorder.finish()
                if recorder.recorder.error:raise OSError('Replay publication failed: '+recorder.recorder.error)
                self.statistics[slot]['completed'].append(str(recorder.recorder.path))
        return rows

    def recording_summary(self,payload=None):return dict(self.statistics)

    def close_recordings(self):
        for recorder in self.recorders.values():
            writer=recorder.recorder
            if not writer._closed:
                writer._stream.flush();writer._stream.close();writer._closed=True
                # Preserve partial .writing files, never publish a false match result.


def live_worker(connection,options):
    bank=None
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        bank=RecordingBank(**options);connection.send(('ok',bank.initial()))
        while True:
            command,payload=connection.recv()
            if command=='close':break
            if command not in ('step','snapshot','restore','recording_summary'):raise ValueError('Unknown live command')
            result=bank.snapshot() if command=='snapshot' else getattr(bank,command)(payload)
            connection.send(('ok',result))
    except BaseException as error:
        try:connection.send(('error',(type(error).__name__,str(error))))
        except (BrokenPipeError,EOFError):pass
    finally:
        if bank is not None:bank.close_recordings()
        connection.close()


class LiveCollector(Collector):
    def __init__(self,workers=4,streams=8,arm='original',partition='engineering',initialization=17,
                 deadline=None,lobby=None,on_spawn=None,recording=None):
        self.recording=recording;self.completed=0;self.latest=None;self.policy_version=0
        if recording is None:
            super().__init__(workers,streams,arm,partition,initialization,deadline,lobby,on_spawn)
            return
        if workers not in (1,2,4,8) or streams<workers or streams%workers:raise ValueError('Invalid worker topology')
        # Same c18ff5f constructor/seed assignments; only bank and worker factories differ.
        self.workers,self.streams=workers,streams;self.arm,self.partition=arm,partition
        self.lobby=lobby or LobbySettings();self.deadline=deadline or perf_counter()+180
        self.connections,self.processes=[],[];self.bank=None;self.generators=[]
        for slot in range(streams):
            seed=int.from_bytes(hashlib.sha256(f'v2-actions:{initialization}:{slot}'.encode()).digest()[:8],'big')
            self.generators.append(torch.Generator().manual_seed(seed))
        self.traces=[hashlib.sha256() for _ in range(streams)]
        options=dict(streams=streams,arm=arm,partition=partition,deadline=self.deadline,lobby=self.lobby,recording=recording)
        try:
            if workers==1:
                self.bank=RecordingBank(slots=list(range(streams)),**options);self.packets=self.bank.initial()
            else:
                ctx=mp.get_context('spawn')
                for rank in range(workers):
                    parent,child=ctx.Pipe()
                    process=ctx.Process(target=live_worker,args=(child,dict(slots=list(range(rank,streams,workers)),**options)))
                    process.start()
                    if on_spawn:on_spawn(process.pid)
                    child.close();self.connections.append(parent);self.processes.append(process)
                self.packets={}
                for connection in self.connections:self.packets.update(self._receive(connection))
        except BaseException:
            self.close();raise

    def collect(self,policy,*args,**kwargs):
        if self.recording is None:return super().collect(policy,*args,**kwargs)
        def observe(module,inputs,output):
            distribution,values=output
            self.latest=(distribution.probs.detach(),values.detach(),distribution.entropy().detach())
        handle=policy.register_forward_hook(observe)
        try:return super().collect(policy,*args,**kwargs)
        finally:handle.remove()

    def _command(self,command,payload=None):
        if command=='step' and self.recording is not None:
            probs,values,entropy=self.latest
            if len(probs)!=self.streams:raise ValueError('Neural telemetry batch mismatch')
            payload={slot:dict(action=action,probabilities=probs[slot].tolist(),value=float(values[slot]),
                              entropy=float(entropy[slot]),training_step=self.completed+slot+1,
                              policy_update=self.policy_version) if slot in self.recording['streams'] else action
                     for slot,action in payload.items()}
            result=super()._command(command,payload);self.completed+=self.streams
            return result
        return super()._command(command,payload)

    def close(self):
        if self.bank is not None and hasattr(self.bank,'close_recordings'):self.bank.close_recordings()
        super().close()
