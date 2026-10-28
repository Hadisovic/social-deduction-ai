"""Observational adapter to the existing completed-match spectator-v1.0 pipeline."""
from time import perf_counter
from spectator.recorder import MatchRecorder
from spectator.replay_format import ReplayReward


class Recorder:
    def __init__(self,match,path,metadata):
        self.match=match;self.seconds=0.;self.reward_cursor=0;self.cumulative={}
        self.recorder=MatchRecorder(match,path,training_session=metadata.get('session','patrick-jane-engineering'),
            game_schema='patrick-jane-v1',
            checkpoint_path=metadata.get('checkpoint',''),model_architecture='Patrick Jane: recurrent PPO, shared GRU256, five heads',
            source_version=metadata.get('source','working-tree'),controller_types={pid:'patrick-jane-ppo' if pid in match.crew else 'scripted-impostor' for pid in match.game.players},
            metadata=dict(metadata,patrick_jane_schema='patrick-jane-v1',population=match.game.experiment.to_dict()['population']))
        self.recorder.recorder.write_event(dict(tick=0,time=0.,kind='patrick_jane_session',channel='spectator',**metadata))

    def neural(self,pid,record,packet,nav):
        started=perf_counter()
        c=record['choice'];names=('interaction','navigation','communication','voting','target')
        probabilities={}
        for name,values in record.get('probabilities',{}).items():
            for i,p in enumerate(values):
                if p:
                    label=(packet.candidates[i].kind+':'+packet.candidates[i].key) if name=='target' and i<len(packet.candidates) else str(i)
                    probabilities[name+'/'+label]=float(p)
        target=packet.candidates[c.target].key if c.target>=0 else packet.other_ids[c.voting] if 0<=c.voting<14 else None
        kind='vote' if packet.phase==2 and c.voting!=14 else 'skip' if packet.phase==2 else 'communication' if packet.phase==1 else 'navigation' if c.interaction==0 else ('','complete_task','report','repair','call_meeting')[c.interaction]
        self.recorder.record_neural(pid,kind,target,probabilities=probabilities,
            value_estimate=record['value'],entropy=record['entropy'],policy_update=record.get('update',0),
            training_step=record.get('training_step',0),legal_actions=list(probabilities),
            selected_action_index=getattr(c,'voting') if packet.phase==2 else None)
        game=self.match.game
        self.recorder.recorder.write_event(dict(tick=game.tick,time=game.time,kind='patrick_jane_decision',
            channel='spectator-private',player_id=pid,heads={name:getattr(c,name) for name in names},
            destination=nav.goal.key if nav.goal else None,route_before_action=list(game.players[pid].navigator.route),
            conditional_head_probabilities=record.get('probabilities',{}),update=record.get('update',0)))
        self.seconds+=perf_counter()-started

    def step(self):
        started=perf_counter();self.recorder.after_step()
        game=self.match.game
        if game.tick%round(game.config.decision_interval/game.config.dt)==1:
            for pid,nav in self.match.nav.items():
                self.recorder.recorder.write_event(dict(tick=game.tick,time=game.time,kind='patrick_jane_route',
                    channel='spectator-private',player_id=pid,destination=nav.goal.key if nav.goal else None,
                    route=list(game.players[pid].navigator.route)))
        self.seconds+=perf_counter()-started

    def reward(self,time,pid,components):
        started=perf_counter()
        value=sum(components.values());self.cumulative[pid]=self.cumulative.get(pid,0.)+value
        self.recorder.recorder.write_reward(ReplayReward(tick=self.match.game.tick,sim_time=time,
            player_id=pid,reward=value,cumulative_reward=self.cumulative[pid],components=components))
        self.seconds+=perf_counter()-started

    def finish(self):
        started=perf_counter();self.recorder.finish();self.seconds+=perf_counter()-started
