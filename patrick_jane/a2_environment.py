"""Versioned classic Skeld additions; public inputs remain role-local values."""
from dataclasses import replace
from collections import Counter
import math
from shapely.geometry import Point as ShapePoint,Polygon
from social_deduction.actor import Role,Phase,Point
from social_deduction.phase3_api import Intent,IntentKind,legal_actions
from social_deduction.phase3_observation import visible_between
from navigation_service import NavStatus
from .environment import PatrickJaneGame
from .api import Action,Event
from .a2_geometry import geometry,planner,door_groups,vents


class A2Game(PatrickJaneGame):
    def reset(self,seed=0):
        super().reset(seed)
        self.base_map=self.map;self.door_groups=door_groups();self.vents=vents()
        self.door_until={};self.door_ready={room:self.experiment.sabotage.initial_cooldown for room in self.door_groups};self.door_pending=set()
        self.vented={};self.vent_ready={};self.vent_transition={};self.vent_direct={pid:[] for pid in self.players}
        self.door_known={pid:set() for pid in self.players};self.door_waiting=set();self.door_local_planners={}
        self.door_seen_time={pid:{} for pid in self.players}
        self.accepted_ballots=[];self.ballot_keys=set();self.explicit_ballots=set();self.a2_metrics=Counter()
        self.communication_counts=Counter();self.opponent_visited={pid:set() for pid in self.players}
        self.deferred_locations={}
        self.direct_vent_evidence={pid:set() for pid in self.players};self.direct_kill_evidence={pid:set() for pid in self.players}
        return self

    @property
    def classic(self):return self.experiment.mechanics_regime=='classic-skeld-a2-v1'

    def observe(self,pid):
        obs=super().observe(pid)
        if not hasattr(self,'accepted_ballots'):return obs
        # Policy masks exclude self. Keep legal targets and reward denominators identical.
        actions=tuple(a for a in obs.actions if not (a.kind is IntentKind.VOTE and a.target==pid))
        if self.classic:
            actions=tuple(a for a in actions if a.target not in self.vented or a.kind not in (IntentKind.KILL,IntentKind.GO_TO_BODY))
            visible=tuple(p for p in obs.visible_players if p.player_id not in self.vented)
            evidence=tuple(e for e in obs.evidence if not (type(e.payload).__name__=='PlayerSeen' and e.payload.player_id in self.vented))
            if pid in self.vented:
                actions=();visible=();evidence=tuple(e for e in evidence if e.provenance.value!='direct')
                obs=replace(obs,visible_bodies=(),interactable_tasks=(),interactable_destinations=())
            elif self.time<self.vent_transition.get(pid,0.)-1e-9:actions=()
            obs=replace(obs,visible_players=visible,evidence=evidence)
        return replace(obs,actions=actions,navigation_status='moving' if pid in self.door_waiting else obs.navigation_status)

    def observe_pj(self,pid):
        obs=super().observe_pj(pid)
        if not hasattr(self,'vent_direct'):return obs
        p=self.players[pid];imp=p.active and p.role is Role.IMPOSTOR;roam=self.phase is Phase.ROAMING
        inside=self.vented.get(pid);ready=self.vent_transition.get(pid,0.)<=self.time+1e-9
        entry=tuple(k for k,v in self.vents.items() if math.dist(p.position,v['position'])<=self.experiment.vent_range
                    and visible_between(self.map,p.position,v['position'],self.experiment.vent_range)) if imp and roam and not inside else ()
        door_targets=tuple(r for r in self.door_groups if self.can_close(pid,r)) if self.classic else ()
        extra=tuple(self.vent_direct[pid])
        if pid in self.door_waiting:extra+=(Event('door_obstruction',pid,room=p.motion.map.region(p.position),position=p.position,observed=self.time,delivered=self.time,source=pid,provenance='direct'),)
        # Closed doors are observed locally; timers and unseen states stay spectator-only.
        actor=obs.actor
        return replace(obs,events=(tuple(e for e in obs.events if e.provenance!='direct') if inside else obs.events)+extra,can_sabotage=obs.can_sabotage and inside is None and not self.door_until,
            vent_state=inside,vent_actions=entry if self.classic else (),
            vent_links=self.vents[inside]['links'] if inside and ready else (),vent_can_exit=bool(inside and ready),
            door_actions=door_targets,door_blocked=pid in self.door_waiting,
            feature_schema='patrick-jane-events-a2-v1',vent_transition_ready=ready)

    def can_close(self,pid,room):
        p=self.players[pid]
        return self.classic and p.active and p.role is Role.IMPOSTOR and pid not in self.vented and self.phase is Phase.ROAMING and self.sabotage is None and room in self.door_groups and self.time>=self.door_ready[room] and not any(d in self.door_until for d in self.door_groups[room])

    def close_doors(self,pid,room):
        if not self.can_close(pid,room):return False
        ids=self.door_groups[room]
        self.door_ready[room]=self.time+self.experiment.doors_cooldown
        for key in ids:
            self.door_until[key]=self.time+self.experiment.doors_duration
            self.door_pending.add(key)
        self.a2_metrics['door_sabotages_accepted']+=1
        own_obs=self.observe_pj(pid)
        self.a2_metrics['door_closures_with_visible_teammate']+=any(p.player_id in own_obs.teammates for p in own_obs.actor.visible_players)
        self._record('doors_closed',player_id=pid,room=room,doors=ids,reopens_at=self.time+self.experiment.doors_duration)
        self._update_doors()
        return True

    def _update_doors(self):
        before=self.map.closed_doors
        had_closed=bool(self.door_until)
        for key,end in list(self.door_until.items()):
            if self.time>=end-1e-9:
                self.door_until.pop(key);self.door_pending.discard(key)
                self._record('door_opened',door_id=key,reason='timer')
        # Avoid embedding an actor in a closing collider: hold that leaf open
        # until clear, without displacing a player or extending its deadline.
        active=[]
        by_id={d['id']:d for d in self.base_map.data['doors']}
        for key in self.door_until:
            shape=Polygon(by_id[key]['points']).buffer(self.map.buffer_radius)
            occupied=any(p.active and pid not in self.vented and shape.covers(ShapePoint(p.position)) for pid,p in self.players.items())
            if occupied and key in self.door_pending:continue
            self.door_pending.discard(key);active.append(key)
        self.map=geometry(tuple(sorted(active)))
        self.planner=planner(tuple(sorted(active)))  # Trusted projection/interaction geometry matches physics.
        if had_closed and not self.door_until:self.sabotage_ready=max(self.sabotage_ready,self.time+self.experiment.sabotage.cooldown)
        if self.map.closed_doors!=before:
            for p in self.players.values():p.motion.map=self.map
            self._invalidate()

    def _vent_evidence(self,pid,key,kind):
        p=self.players[pid];point=self.vents[key]['position']
        for observer in self.players.values():
            if observer.active and observer.player_id not in self.vented and observer.player_id!=pid and visible_between(self.map,observer.position,point,self.settings_for(observer.role).sight_range):
                event=Event(kind,pid,key,self.map.region(point),point,self.time,self.time,observer.player_id,'direct')
                self.vent_direct[observer.player_id].append(event);self.direct_vent_evidence[observer.player_id].add(pid)
                if observer.role is Role.CREWMATE:self.a2_metrics['crew_witnessed_'+kind]+=1
                self._record(kind+'_witnessed',channel='direct',observer_id=observer.player_id,player_id=pid,vent_id=key)

    def vent_action(self,pid,kind,key):
        if not self.classic or pid not in self.players:return False
        p=self.players[pid]
        if not p.active or p.role is not Role.IMPOSTOR or self.phase is not Phase.ROAMING:return False
        inside=self.vented.get(pid)
        if self.time<self.vent_transition.get(pid,0.)-1e-9:return False
        if kind=='vent_enter':
            if inside or key not in self.vents or math.dist(p.position,self.vents[key]['position'])>self.experiment.vent_range or not visible_between(self.map,p.position,self.vents[key]['position'],self.experiment.vent_range):return False
            self._vent_evidence(pid,key,kind);self._interrupt(p,'vent_enter');self.vented[pid]=key
        elif kind=='vent_move':
            if inside is None or key not in self.vents[inside]['links']:return False
            self.vented[pid]=key
        elif kind=='vent_exit':
            if inside is None or key!=inside or not self.map.contains(self.vents[key]['position']):return False
            p.motion.spawn(self.vents[key]['position']);self.vented.pop(pid);self._vent_evidence(pid,key,kind)
        else:return False
        p.intention=kind+':'+key;self.vent_transition[pid]=self.time+self.experiment.vent_transition_seconds
        self.a2_metrics[kind]+=1;self._record(kind,player_id=pid,vent_id=key,position=self.vents[key]['position'])
        self._invalidate();return True

    def _start_meeting(self,reporter,body=None):
        if hasattr(self,'vented'):
            # Meeting removes vent occupancy without inventing visible vent exits.
            for pid,key in list(self.vented.items()):
                self.players[pid].motion.spawn(self.vents[key]['position']);self._record('vent_meeting_reset',player_id=pid,vent_id=key)
            self.vented.clear();self.vent_transition.clear()
            if self.door_until:
                self._record('doors_opened',reason='meeting',doors=sorted(self.door_until))
                self.sabotage_ready=max(self.sabotage_ready,self.time+self.experiment.sabotage.cooldown)
            self.door_until.clear();self.door_pending.clear();self.map=self.base_map;self.planner=planner(())
            for p in self.players.values():p.motion.map=self.map
            self.door_waiting.clear()
        return super()._start_meeting(reporter,body)

    def _vote(self,pid,target):
        eligible=tuple(sorted(self._eligible_voters()-{pid}));key=(self.meeting_id,pid)
        if key in self.ballot_keys or target is not None and target not in eligible:return False
        if not super()._vote(pid,target):return False
        n=len(eligible);imps=sum(self.players[t].role is Role.IMPOSTOR for t in eligible);p=imps/n if n else 0.
        correct=target is not None and self.players[target].role is Role.IMPOSTOR
        amount=.2*(float(correct)-p) if target is not None else 0.
        self.ballot_keys.add(key)
        ballot=dict(meeting_id=self.meeting_id,voter_id=pid,target_id=target,eligible_targets=eligible,
            voter_role=self.players[pid].role.value,
            impostor_candidates=imps,candidate_count=n,base_rate=p,correct=correct,reward=amount,
            explicit=(pid in self.explicit_ballots),direct_vent_evidence=target in self.direct_vent_evidence[pid],
            direct_kill_evidence=target in self.direct_kill_evidence[pid],communication_before_ballot=self.communication_counts[self.meeting_id])
        self.accepted_ballots.append(ballot)
        # Offline/debug only. Never appended to public/direct actor channels.
        self._record('a2_ballot_credit',channel='training-private',**ballot)
        return True

    def _resolve_meeting(self):
        counts=Counter(self.votes.values());highest=max(counts.values(),default=0)
        self.a2_metrics['vote_ties']+=sum(v==highest for v in counts.values())>1
        self.a2_metrics['no_ejection_meetings']+=self.tally_votes(self.votes) is None
        super()._resolve_meeting()

    def _advance_physics(self):
        if not self.classic:return super()._advance_physics()
        # A map-floor location remains a legal intention while temporarily
        # covered by a door. Apply after report/kill precedence, before motion.
        from navigation_service import NavTarget
        for pid,point in self.deferred_locations.items():
            if self.players[pid].active:self._navigate(self.players[pid],NavTarget.location(point),'location')
        self.deferred_locations={}
        by_id={d['id']:d for d in self.base_map.data['doors']}
        for pid,p in self.players.items():
            if not p.active:continue
            if pid in self.vented:p.velocity=(0.,0.);continue  # Kill cooldown pauses underground.
            p.cooldown=round(max(0.,p.cooldown-self.config.dt),9)
            old=p.position;known=self.door_known[pid]
            for key,d in by_id.items():
                center=Polygon(d['points']).centroid
                if self._sees_door(p,d):
                    if key in self.map.closed_doors:known.add(key);self.door_seen_time[pid][key]=self.time
                    else:known.discard(key)
                # Direct contact with own next motion segment is legitimate obstruction feedback.
                if key in self.map.closed_doors and p.navigator.status is NavStatus.MOVING and p.navigator.route:
                    waypoint=p.navigator.route[min(getattr(p.navigator,'_waypoint',1),len(p.navigator.route)-1)]
                    delta=math.dist(old,waypoint);fraction=min(1.,self.config.speed*self.config.dt/max(delta,1e-9))
                    near=tuple(a+(b-a)*fraction for a,b in zip(old,waypoint))
                    from shapely.geometry import LineString
                    if Polygon(d['points']).buffer(self.map.buffer_radius).intersects(LineString([old,near])):known.add(key);self.door_seen_time[pid][key]=self.time
            # Forget a reopened obstruction only once local sight/route contact
            # permits observing it. No global door timer reaches a policy.
            for key in tuple(known):
                # A remembered obstruction expires after the public maximum
                # closure duration. No remaining hidden door timer is consulted.
                if self.time-self.door_seen_time[pid].get(key,self.time)>=self.experiment.doors_duration:
                    known.discard(key);self.a2_metrics['stale_door_knowledge_expired']+=1;continue
                if key not in self.map.closed_doors:
                    center=Polygon(by_id[key]['points']).centroid
                    if self._sees_door(p,by_id[key]):known.discard(key)
            revision=tuple(sorted(known))
            if self.door_local_planners.get(pid)!=revision:
                self.door_local_planners[pid]=revision;p.navigator.planner=planner(revision)
                if p.navigator.target is not None and p.navigator.status in (NavStatus.MOVING,NavStatus.NO_ROUTE,NavStatus.STUCK,NavStatus.INVALID_DESTINATION):p.navigator._plan(old)
                self.a2_metrics['door_route_invalidations']+=1
            if p.navigator.status in (NavStatus.NO_ROUTE,NavStatus.INVALID_DESTINATION) and known:
                if pid not in self.door_waiting:self.a2_metrics['door_navigation_waits']+=1
                self.door_waiting.add(pid);p.velocity=(0.,0.);self.a2_metrics['door_wait_actor_seconds']+=self.config.dt;continue
            if pid in self.door_waiting:
                self.door_waiting.discard(pid);self.a2_metrics['door_route_recoveries']+=1
            displacement=p.navigator.command(old,self.config.dt)
            if p.navigator.awaiting_feedback:
                new,hit=p.motion.advance_motion(displacement,self.config.dt);p.navigator.feedback(new,hit)
            p.velocity=tuple((a-b)/self.config.dt for a,b in zip(p.position,old))

    def _sees_door(self,p,door):
        from shapely.ops import nearest_points
        shape=Polygon(door['points']);near=nearest_points(ShapePoint(p.position),shape)[1]
        target=(near.x,near.y);distance=math.dist(p.position,target)
        if distance>self.settings_for(p.role).sight_range:return False
        fraction=max(0.,(distance-.02)/max(distance,1e-9))
        approach=tuple(a+(b-a)*fraction for a,b in zip(p.position,target))
        return visible_between(self.map,p.position,approach,self.settings_for(p.role).sight_range)

    def _navigate(self,p,target,description):
        # Retain a legal intention even when a known temporary door has no route.
        super()._navigate(p,target,description)
        if self.classic and p.navigator.status in (NavStatus.NO_ROUTE,NavStatus.INVALID_DESTINATION) and self.door_known[p.player_id]:
            self.door_waiting.add(p.player_id)
            if self.event_log and self.event_log[-1]['kind']=='navigation_failed':self.event_log.pop();self.metrics['navigation_failures']-=1

    def _eliminate(self,killer,victim):
        if not self.classic:return super()._eliminate(killer,victim)
        from phase3_engine import Body
        from social_deduction.actor import Provenance
        from social_deduction.phase3_api import EventView,WitnessedElimination
        for observer in self.players.values():
            if observer.active and observer.player_id not in self.vented and visible_between(self.map,observer.position,killer.position,self.settings_for(observer.role).sight_range) and visible_between(self.map,observer.position,victim.position,self.settings_for(observer.role).sight_range):
                self.direct_kill_evidence[observer.player_id].add(killer.player_id)
                if observer.role is Role.CREWMATE:self.a2_metrics['crew_kill_witness_events']+=1
                self.direct_events[observer.player_id].append(EventView(self.tick,Provenance.DIRECT,WitnessedElimination(killer.player_id,victim.player_id,Point(*victim.position),self.map.region(victim.position))))
                self._record('elimination_witnessed',channel='direct',observer_id=observer.player_id,killer_id=killer.player_id,victim_id=victim.player_id)
        self._interrupt(victim,'eliminated');victim.status='dead'
        self.bodies[victim.player_id]=Body(victim.player_id,victim.position,self.tick,killer.player_id)
        self._interrupt(killer,'kill_lunge');killer.motion.spawn(victim.position);killer.cooldown=self.config.kill_cooldown
        self.metrics['eliminations']+=1;self._record('elimination',killer_id=killer.player_id,victim_id=victim.player_id,position=victim.position)
        self._make_ghost(victim);self._invalidate();self._check_win()

    def _completed(self,player,task,*,ghost=False):
        cursors={pid:len(self.v2_direct[pid]) for pid in getattr(self,'vented',{})}
        super()._completed(player,task,ghost=ghost)
        # Underground scripts cannot gain delayed visual-task proof after exit.
        for pid,start in cursors.items():del self.v2_direct[pid][start:]

    def step(self,actions=None):
        if self.is_terminal:return self.result
        self.explicit_ballots={pid for pid,a in (actions or {}).items() if type(a) is Intent and a.kind in (IntentKind.VOTE,IntentKind.SKIP)}
        if not self.classic:return super().step(actions)
        self._update_doors();ordinary={};self.deferred_locations={}
        for pid,a in sorted((actions or {}).items()):
            if type(a) is Action and a.kind in ('doors','vent_enter','vent_move','vent_exit'):
                if a.kind=='doors':
                    self.a2_metrics['door_sabotages_selected']+=1;ok=self.close_doors(pid,a.target)
                else:ok=self.vent_action(pid,a.kind,a.target)
                if not ok:self.metrics['illegal_actions']+=1;self._record('a2_action_rejected',player_id=pid,action=a.kind)
            elif pid in self.vented and a!=Intent(IntentKind.WAIT):self.metrics['illegal_actions']+=1
            elif type(a) is Intent and a.kind is IntentKind.GO_TO_LOCATION and a.position is not None and self.phase is Phase.ROAMING and self.players[pid].active and not self.time<self.vent_transition.get(pid,0.) and self.base_map.contains((a.position.x,a.position.y)) and not self.map.contains((a.position.x,a.position.y)):
                self.deferred_locations[pid]=(float(a.position.x),float(a.position.y))
            else:ordinary[pid]=a
        cursor=len(self.event_log);meeting=self.meeting_id
        result=super().step(ordinary)
        self.communication_counts[meeting]+=sum(e['kind'] in ('structured_claim','claim') for e in self.event_log[cursor:])
        for pid,p in self.players.items():
            if p.role is Role.IMPOSTOR and p.active and pid not in self.vented:
                self.opponent_visited[pid].add((round(p.position[0]),round(p.position[1])))
        self.a2_metrics['opponent_visited_cells']=sum(len(v) for v in self.opponent_visited.values())
        self.a2_metrics['impostor_self_reports']+=sum(e['kind']=='report' and self.players[e['reporter_id']].role is Role.IMPOSTOR for e in self.event_log[cursor:])
        self.a2_metrics['closed_door_seconds']+=self.config.dt*len(self.map.closed_doors)
        self._update_doors()
        return result
