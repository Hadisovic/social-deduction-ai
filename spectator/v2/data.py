"""Chronological recorded-data projections. Truth is used only for spectator QA."""
from bisect import bisect_right
from collections import Counter,defaultdict
from copy import copy
import math
import re

from spectator.playback import ReplayPlayer, EventMarker

MISSING='Not recorded'


def time_of(record):return float(record.get('sim_time',record.get('time',0.)))


def family(label):
    name=re.sub(r'^\d+:\s*','',str(label)).lower().split(':')[0].split(' ')[0]
    if name in ('room','task','location','body','follow','go_to_room','go_to_task','go_to_location','go_to_body'):return 'Travel'
    if name in ('vote','skip'):return 'Voting'
    if name in ('claim','claim_suspect','claim_defend','suspect','defend','claim_task'):return 'Claims'
    if name=='report':return 'Reporting'
    if name in ('do_task','interact','complete_task'):return 'Tasks'
    if name=='kill':return 'Kill'
    return name.replace('_',' ').title()


def event_actor(event):
    return next((event[k] for k in ('player_id','speaker_id','voter_id','reporter_id','killer_id','caller_id') if event.get(k)),None)


def event_text(event,debug=False):
    kind=event.get('kind','event');actor=event_actor(event) or 'System'
    if kind=='a2_ballot_credit':return f"[TRAINING DEBUG] {actor} ballot reward {event.get('reward',0):+.5f}; p={event.get('base_rate',0):.3f}"
    if kind.startswith('vent_'):return f"{actor} · {kind.replace('_',' ')} · vent {event.get('vent_id','')}"
    if kind in ('doors_closed','doors_opened','door_opened'):return f"{actor} · {kind.replace('_',' ')} {event.get('room',event.get('door_id',''))}"
    if kind in ('claim','structured_claim'):
        return f"{actor} · {str(event.get('claim_kind','claim')).replace('_',' ')} {event.get('subject_id','')}".strip()
    if kind=='vote_submitted' and not debug:return f'{actor} · ballot submitted (target hidden)'
    if kind in ('vote','vote_submitted'):
        target=event.get('target_id',MISSING)
        return f"{actor} · {'SKIP' if target is None else 'vote '+str(target)}"
    if kind=='report':return f"{actor} · reported {event.get('victim_id','body')}"
    if kind in ('kill','elimination'):return f"{actor} · elimination {event.get('victim_id','')}"
    if kind=='task_completed':return f"{actor} · completed {event.get('task_id','task')}"
    return f"{actor} · {kind.replace('_',' ')} {event.get('player_id','') if kind=='ejection' else ''}".strip()


def category(kind):
    if kind in ('kill','elimination'):return 'Kills'
    if kind in ('meeting_started','voting_started','exploration_resumed','meeting_resolution','call_meeting','meeting_called') or 'emergency' in kind:return 'Meetings'
    if kind in ('vote','vote_submitted','ejection'):return 'Votes'
    if kind in ('claim','structured_claim'):return 'Claims'
    if kind=='report':return 'Reports'
    if kind=='task_completed':return 'Tasks'
    if kind=='reward':return 'Rewards'
    if kind=='result':return 'Result'
    return 'Other'


class ReplayIndex:
    def __init__(self,data):
        self.issues=[];data=dict(data)
        for key in ('events','neural','rewards'):
            good=[]
            for record in data.get(key,[]):
                try:
                    valid=isinstance(record,dict) and isinstance(record.get('sim_time',record.get('time')),(int,float)) and math.isfinite(time_of(record)) and time_of(record)>=0
                    if key=='neural' and valid:
                        for field in ('action_probabilities','suspicion_probabilities'):
                            values=record.get(field,{})
                            valid=valid and isinstance(values,dict) and all(isinstance(p,(int,float)) and math.isfinite(p) and 0<=p<=1 for p in values.values())
                    if key=='rewards' and valid:
                        valid=isinstance(record.get('components',{}),dict) and all(isinstance(v,(int,float)) and math.isfinite(v) for v in record.get('components',{}).values())
                    if valid:good.append(record)
                    else:self.issues.append('Invalid '+key+' record ignored in display')
                except (ValueError,TypeError,AttributeError):self.issues.append('Invalid '+key+' record ignored in display')
            data[key]=good
        self.data=data;self.player=ReplayPlayer(data);self.header=data.get('header',{})
        self.config=self.header.get('config',{});self.ticks=data.get('ticks',[])
        self.tick_times=[time_of(x) for x in self.ticks]
        self.events=sorted(data.get('events',[]),key=time_of);self.event_times=[time_of(e) for e in self.events]
        self.a2_states=[e for e in self.events if e.get('kind')=='a2_world_state'];self.a2_times=[time_of(e) for e in self.a2_states]
        self.neurals=defaultdict(list);self.neural_times={};self.rewards=defaultdict(list);self.reward_times={};self.reward_sums={}
        for n in sorted(data.get('neural',[]),key=time_of):self.neurals[n.get('player_id','')].append(n)
        for r in sorted(data.get('rewards',[]),key=time_of):self.rewards[r.get('player_id','')].append(r)
        for pid,records in self.neurals.items():self.neural_times[pid]=[time_of(r) for r in records]
        for pid,records in self.rewards.items():
            self.reward_times[pid]=[time_of(r) for r in records];total=Counter();sums=[]
            for r in records:
                total.update({k:v for k,v in r.get('components',{}).items() if isinstance(v,(int,float))});sums.append(dict(total))
            self.reward_sums[pid]=sums
        self._cached_tick=None;self._frame=None
        self._build_markers()

    def _build_markers(self):
        self.markers=[]
        colors={'Kills':(249,113,121),'Reports':(185,245,106),'Meetings':(118,189,249),'Votes':(216,171,255),'Claims':(105,215,209),'Tasks':(249,204,116),'Rewards':(155,182,215),'Result':(249,243,165)}
        for e in self.events:
            group=category(e.get('kind',''))
            if group!='Other':
                marker=EventMarker(time_of(e),e.get('tick',0),e.get('kind',''),event_text(e),colors.get(group,(150,160,180)),event_actor(e))
                marker.channel=e.get('channel','unrecorded');self.markers.append(marker)
        for r in self.data.get('rewards',[]):
            marker=EventMarker(time_of(r),r.get('tick',0),'reward','Recorded reward',colors['Rewards'],r.get('player_id'))
            marker.meaningful=any(k!='time' and v for k,v in r.get('components',{}).items());self.markers.append(marker)
        result=self.data.get('result')
        if result:self.markers.append(EventMarker(time_of(result),result.get('tick',0),'result',result.get('reason','Match end'),colors['Result']))
        self.markers.sort(key=lambda m:m.sim_time)
        self.player.markers=[m for m in self.markers if m.kind!='reward']

    def frame(self):
        tick=self.player._current_tick_index
        if tick!=self._cached_tick:
            self._frame=self.player.current_state();self._cached_tick=tick
        return self._frame

    def a2_state(self):
        i=bisect_right(self.a2_times,self.player.current_sim_time)-1
        return self.a2_states[i] if i>=0 else None

    def raw(self,pid,t=None):
        if not self.ticks:return {}
        when=self.frame().sim_time if t is None else t
        i=max(0,bisect_right(self.tick_times,when+1e-8)-1)
        return self.ticks[i].get('players',{}).get(pid,{})

    def role(self,pid):return self.raw(pid).get('role') or self.header.get('agent_roles',{}).get(pid)

    def before(self,records,times,when):return records[:bisect_right(times,when+1e-8)]

    def history(self,pid,t=None):
        return self.before(self.neurals.get(pid,[]),self.neural_times.get(pid,[]),self.frame().sim_time if t is None else t)

    def neural(self,pid,t=None):
        records=self.history(pid,t);return records[-1] if records else None

    def events_until(self,t=None):
        return self.before(self.events,self.event_times,self.frame().sim_time if t is None else t)

    def task_progress(self,pid):
        r=self.raw(pid);done,total=r.get('tasks_completed'),r.get('tasks_total')
        if not isinstance(done,(int,float)) or not isinstance(total,(int,float)) or total<=0:return None
        return dict(completed=done,total=total,fraction=max(0.,min(1.,done/total)),stage=r.get('task_progress'),active=r.get('interaction_task'))

    def sabotage(self):
        state=None
        for event in self.events_until():
            kind=event.get('kind')
            if kind in ('sabotage_repaired','sabotage_cancelled'):state=None
            elif kind in ('sabotage_started','live_sabotage'):
                name=event.get('sabotage')
                deadline=event.get('deadline')
                state=dict(kind=name,remaining=max(0,deadline-self.frame().sim_time) if isinstance(deadline,(int,float)) else None) if name else None
        return state

    def team_tasks(self):
        ids=[p for p in self.frame().players if self.role(p)=='crewmate']
        values=[self.task_progress(p) for p in ids]
        if not ids or any(v is None for v in values):return None
        done=sum(v['completed'] for v in values);total=sum(v['total'] for v in values)
        return dict(completed=done,total=total,fraction=done/total)

    def reward(self,pid):
        i=bisect_right(self.reward_times.get(pid,[]),self.frame().sim_time+1e-8)-1
        if i<0:return None
        latest=self.rewards[pid][i]
        components=self.reward_sums[pid][i]
        cumulative=latest.get('cumulative_reward')
        balance=cumulative-sum(components.values()) if isinstance(cumulative,(int,float)) else None
        return dict(latest=latest,components=components,history=self.rewards[pid][:i+1],unattributed_balance=balance)

    def actor_stats(self,pid):
        c=Counter()
        for e in self.events_until():
            k=e.get('kind')
            if k in ('kill','elimination') and e.get('killer_id')==pid:c['kills']+=1
            if k=='report' and e.get('reporter_id')==pid:c['reports']+=1
            if k=='task_completed' and e.get('player_id')==pid:
                c['tasks']+=1
                if self.raw(pid,time_of(e)).get('status') in ('dead','ejected'):c['ghost_tasks']+=1
            if k=='vote_submitted' and e.get('voter_id')==pid:c['ballots']+=1
        return c

    def meeting(self,omniscient=False,reveal=False):
        events=self.events_until();start=next((e for e in reversed(events) if e.get('kind')=='meeting_started'),None)
        if start is None:return None
        mid=start.get('meeting_id');when=time_of(start)
        eligible=[p for p in start.get('participants',[]) if self.raw(p,when).get('status','alive')=='alive']
        scoped=[e for e in events if time_of(e)>=when and e.get('meeting_id') in (None,mid)]
        published={e.get('voter_id'):e for e in scoped if e.get('kind')=='vote' and e.get('channel')=='public'}
        submitted={e.get('voter_id'):e for e in scoped if e.get('kind')=='vote_submitted' and e.get('meeting_id')==mid}
        ballots=[]
        for pid in eligible:
            record=published.get(pid) or submitted.get(pid);visible=pid in published or (omniscient and reveal)
            target=record.get('target_id') if record and 'target_id' in record and visible else None
            state='missing' if record is None else 'hidden' if not visible else 'not_recorded' if 'target_id' not in record else ('skip' if pid in submitted else 'published_skip') if target is None else 'vote'
            ballots.append(dict(voter=pid,state=state,target=target,debug=record is not None and visible and pid not in published))
        report=next((e for e in reversed(events) if e.get('kind') in ('report','emergency_called') and 0<=when-time_of(e)<.41),None)
        phase=self.frame().phase
        voting=next((e for e in scoped if e.get('kind')=='voting_started'),None)
        duration=self.config.get('voting_seconds' if phase=='voting' else 'discussion_seconds')
        began=time_of(voting) if phase=='voting' and voting else when
        remaining=max(0,duration-(self.frame().sim_time-began)) if isinstance(duration,(int,float)) and phase in ('discussion','voting') else None
        claims=[e for e in scoped if e.get('kind') in ('claim','structured_claim') and e.get('speaker_id') in eligible]
        return dict(id=mid,started=when,eligible=eligible,ballots=ballots,claims=claims,report=report,remaining=remaining,events=scoped)

    def evidence(self,pid,when=None):
        t=self.frame().sim_time if when is None else when
        public=[e for e in self.events_until(t) if e.get('channel')=='public']
        n=self.neural(pid,t);features=n.get('observation_features') if n else None
        witnessed=[e for e in self.events_until(t) if e.get('channel')=='direct' and e.get('observer_id')==pid and e.get('kind') in ('vent_enter_witnessed','vent_exit_witnessed','elimination_witnessed')]
        return dict(public=public,private=features if features else None,witnessed=witnessed)

    def decision_delta(self,pid,a,b):
        if time_of(a)>time_of(b):a,b=b,a
        pa=a.get('action_probabilities',{});pb=b.get('action_probabilities',{})
        keys=set(pa)|set(pb)
        tv=.5*sum(abs(pa.get(k,0)-pb.get(k,0)) for k in keys)
        sa=a.get('suspicion_probabilities',{});sb=b.get('suspicion_probabilities',{})
        public=[e for e in self.events if time_of(a)<time_of(e)<=time_of(b) and e.get('channel')=='public']
        return dict(start=time_of(a),end=time_of(b),total_variation=tv,
            suspicion_delta={p:sb[p]-sa[p] for p in sa.keys()&sb.keys()},public_events=public,
            private_evidence=MISSING,interpretation='Distribution change is observed; causal reasoning is not recorded.')

    def matching_header(self,other):
        a,b=self.header,other
        if not a.get('training_session') or not b.get('training_session'):return False
        keys=('training_session','scenario_seed','game_schema','source_version')
        if any(a.get(k)!=b.get(k) for k in keys):return False
        ca,cb=a.get('config',{}),b.get('config',{})
        fields=('context','initialization','prefix_handoff_seconds','environment_schema','dt','kill_cooldown','discussion_seconds','voting_seconds')
        if any(ca.get(k)!=cb.get(k) for k in fields):return False
        if ca.get('training_condition')!='zero' and cb.get('training_condition')!='zero' and ca.get('policy_version')!=cb.get('policy_version'):return False
        return ca.get('training_condition')!=cb.get('training_condition')

    def reward_label(self):
        profile=self.config.get('reward_profile')
        if not isinstance(profile,dict):return MISSING
        if profile.get('report')==0 and profile.get('correct_vote')==0:return 'Original / common evaluation scoring'
        return f"Recorded profile: report {profile.get('report',MISSING)}, vote {profile.get('correct_vote',MISSING)}/{profile.get('incorrect_vote',MISSING)}"

    def filtered_events(self,group='All',pid='All',debug=False,query='',reveal_ballots=False):
        result=[]
        for e in self.events_until():
            if not debug and e.get('channel') not in ('public',None,''):continue
            if group!='All' and category(e.get('kind',''))!=group:continue
            if pid!='All' and pid not in (event_actor(e),e.get('victim_id'),e.get('subject_id')):continue
            text=event_text(e,debug and reveal_ballots)
            if query.casefold() not in text.casefold():continue
            result.append(dict(time=time_of(e),text=text,category=category(e.get('kind','')),event=e))
        if group in ('All','Rewards'):
            for owner,records in self.rewards.items():
                if pid!='All' and pid!=owner:continue
                for r in records:
                    if time_of(r)>self.frame().sim_time+1e-8:break
                    if group=='All' and not any(k!='time' and v for k,v in r.get('components',{}).items()):continue
                    value=r.get('reward');label=f'{owner} · recorded reward {value:+.4f}' if isinstance(value,(int,float)) else f'{owner} · recorded reward'
                    if query.casefold() in (label+' '+str(r.get('components',{}))).casefold():result.append(dict(time=time_of(r),text=label,category='Rewards',event=r))
        return sorted(result,key=lambda r:r['time'])
