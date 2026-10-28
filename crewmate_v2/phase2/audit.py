"""Observer-only meters. Predictions/actions first; role scoring only after play."""
from collections import Counter, defaultdict
from dataclasses import asdict
import math
import statistics

from social_deduction.actor import Phase
from social_deduction.phase3_api import IntentKind, intent_allowed


def ratio(a,b):return a/b if b else None


class OpportunityAudit:
    def __init__(self):
        self.pairs={};self.counts=defaultdict(Counter);self.decisions=[];self.events=[]
        self.moves=defaultdict(Counter);self.killed_by={};self.ballot_opportunities={}

    def observe(self,obs):
        if not obs.own.active:return
        pid=obs.own.player_id;counts=self.counts[pid]
        legal={a.target for a in obs.actions if a.kind is IntentKind.REPORT}
        for body in obs.visible_bodies:
            key=(pid,body.victim_id)
            row=self.pairs.setdefault(key,dict(actor=pid,body=body.victim_id,first_visible=obs.time_seconds,
                first_range=None,first_legal=None,visible_ticks=0,range_ticks=0,legal_ticks=0,
                decision_windows=0,submitted=[],accepted=[]))
            row['visible_ticks']+=1
            distance=math.dist((obs.own.position.x,obs.own.position.y),(body.position.x,body.position.y))
            if distance<=obs.settings.report_range+1e-8:
                row['range_ticks']+=1
                if row['first_range'] is None:row['first_range']=obs.time_seconds
            if body.victim_id in legal:
                row['legal_ticks']+=1
                if row['first_legal'] is None:row['first_legal']=obs.time_seconds
        counts['active_ticks']+=1

    def decision(self,obs,intent,*,neural=None):
        pid=obs.own.player_id;c=self.counts[pid];c['decision_windows']+=1
        kinds={a.kind for a in obs.actions}
        c['report_legal_decision_windows']+=IntentKind.REPORT in kinds
        c['kill_legal_decision_windows']+=IntentKind.KILL in kinds
        c['emergency_legal_decision_windows']+=IntentKind.CALL_MEETING in kinds
        c['action_'+intent.kind.value]+=1
        for a in obs.actions:
            if a.kind is IntentKind.REPORT:
                row=self.pairs.get((pid,a.target))
                if row is None:
                    row=dict(actor=pid,body=a.target,first_visible=obs.time_seconds,first_range=obs.time_seconds,
                        first_legal=obs.time_seconds,visible_ticks=0,range_ticks=0,legal_ticks=0,
                        decision_windows=0,submitted=[],accepted=[])
                    self.pairs[(pid,a.target)]=row
                row['decision_windows']+=1
        if intent.kind is not IntentKind.WAIT and not intent_allowed(obs,intent):raise ValueError('Audited illegal intention')
        if intent.kind in (IntentKind.GO_TO_TASK,IntentKind.GO_TO_ROOM,IntentKind.GO_TO_BODY):self.moves[pid][intent.kind.value+':'+str(intent.target)]+=1
        if intent.kind is IntentKind.GO_TO_LOCATION:self.moves[pid]['go_to_location']+=1
        if IntentKind.VOTE in kinds or IntentKind.SKIP in kinds:
            self.ballot_opportunities.setdefault((pid,obs.context.meeting_id),dict(actor=pid,meeting=obs.context.meeting_id,
                reporter=self._reporter(obs),candidates=[x for x in obs.context.participants if x!=pid]))
        if neural is not None:
            self.decisions.append(dict(actor=pid,time=obs.time_seconds,tick=obs.tick,
                observation=asdict(obs),**neural))

    @staticmethod
    def _reporter(obs):
        from social_deduction.actor import Report, Meeting
        meetings=[e for e in obs.evidence if type(e.payload) is Meeting and e.payload.meeting_id==obs.context.meeting_id]
        if not meetings:return None
        reports=[e.payload for e in obs.evidence if type(e.payload) is Report and e.tick==meetings[-1].tick]
        return reports[-1].reporter_id if reports else None

    def before_step(self,game,actions):
        # Trusted observer reads each separate packet. No controller receives this object.
        for pid in sorted(game.players):
            obs=game.observe(pid);self.observe(obs)
            intent=actions.get(pid)
            if intent is not None and intent.kind is IntentKind.REPORT:
                row=self.pairs.get((pid,intent.target))
                if row is None or not intent_allowed(obs,intent):raise ValueError('Report outside legitimate local opportunity')
                row['submitted'].append(game.time)

    def after_step(self,events):
        for e in events:
            self.events.append(dict(e))
            if e['kind']=='elimination':self.killed_by[e['victim_id']]=e['killer_id']
            if e['kind']=='report':
                row=self.pairs.get((e['reporter_id'],e['victim_id']))
                if row is None:raise ValueError('Accepted report without prior legal packet')
                row['accepted'].append(e['time'])

    def summary(self,game):
        roles={pid:p.role.value for pid,p in game.players.items()}
        out={};events=self.events
        for role in ('crewmate','impostor'):
            ids=[pid for pid,r in roles.items() if r==role];pairs=[p for p in self.pairs.values() if p['actor'] in ids]
            counts=Counter()
            for pid in ids:counts.update(self.counts[pid])
            accepted=[e for e in events if e['kind']=='report' and e['reporter_id'] in ids]
            kills=[e for e in events if e['kind']=='elimination' and e['killer_id'] in ids]
            delay=[t-p['first_legal'] for p in pairs for t in p['accepted'] if p['first_legal'] is not None]
            legalpairs=sum(p['first_legal'] is not None for p in pairs)
            out[role]=dict(visible_actor_body_pairs=len(pairs),range_actor_body_pairs=sum(p['first_range'] is not None for p in pairs),
                legal_actor_body_pairs=legalpairs,decision_actor_body_pairs=sum(p['decision_windows']>0 for p in pairs),
                visible_ticks=sum(p['visible_ticks'] for p in pairs),range_ticks=sum(p['range_ticks'] for p in pairs),
                legal_ticks=sum(p['legal_ticks'] for p in pairs),counts=dict(counts),accepted_reports=len(accepted),
                accepted_report_pair_conversion=ratio(len(accepted),legalpairs),
                submitted_report_window_conversion=ratio(counts['action_report'],counts['report_legal_decision_windows']),
                accepted_kills=len(kills),kill_window_conversion=ratio(len(kills),counts['kill_legal_decision_windows']),
                self_reports=sum(self.killed_by.get(e['victim_id'])==e['reporter_id'] for e in accepted),
                first_legal_to_accepted_seconds=delay,
                median_report_delay=statistics.median(delay) if delay else None)
        published=[e for e in events if e['kind']=='vote' and e.get('channel')=='public' and roles.get(e['voter_id'])=='crewmate']
        reporter_scores=[]
        for b in self.ballot_opportunities.values():
            if roles.get(b['actor'])!='crewmate' or b['reporter'] is None or b['actor']==b['reporter']:continue
            candidates=b['candidates'];imps=[p for p in candidates if roles[p]=='impostor']
            if len(imps)!=1:continue
            reporter=b['reporter'];score=float(reporter==imps[0]) if reporter in candidates else 1/len(candidates)
            reporter_scores.append(dict(meeting=b['meeting'],role=roles[reporter],actor=b['actor'],score=score,uniform=1/len(candidates)))
        return dict(by_role=out,meetings=sum(e['kind']=='meeting_started' for e in events),
            correct_crew_ballots=sum(e.get('target_id') is not None and roles.get(e['target_id'])=='impostor' for e in published),
            wrong_crew_ballots=sum(e.get('target_id') is not None and roles.get(e['target_id'])=='crewmate' for e in published),
            published_crew_skips=sum(e.get('target_id') is None for e in published),
            discovered_bodies=len({e['victim_id'] for e in events if e['kind']=='report'}),
            unreported_kills=len(set(self.killed_by)-{e['victim_id'] for e in events if e['kind']=='report'}),
            claim_types=dict(Counter(e.get('claim_kind') for e in events if e['kind']=='claim' and roles.get(e.get('speaker_id'))=='impostor')),
            impostor_victims=dict(Counter(e['victim_id'] for e in events if e['kind']=='elimination')),
            navigation={pid:dict(c) for pid,c in self.moves.items()},reporter_only=reporter_scores)


class AuditedController:
    def __init__(self,controller,audit):self.controller=controller;self.audit=audit
    def decide(self,obs):
        intent=self.controller.decide(obs);self.audit.decision(obs,intent);return intent


def install_observer(match,audit):
    """Wrap an existing step without changing rules, actions or RNG consumption."""
    step=match.game.step
    def observed(actions=None):
        actions=actions or {};audit.before_step(match.game,actions);mark=len(match.game.event_log)
        result=step(actions);audit.after_step(match.game.event_log[mark:]);return result
    match.game.step=observed


def neural_observer(audit):
    def capture(pid,observation,memory,belief,options,probabilities,packet,selected):
        from phase5_features import mechanical_intent
        intent=mechanical_intent(options[selected],observation)
        if intent is None:
            from social_deduction.phase3_api import Intent
            intent=Intent(IntentKind.WAIT)
        audit.decision(observation,intent,neural=dict(selected=selected,
            options=[dict(kind=c.kind,target=c.target_player,label=c.label) for c in options],
            probabilities=probabilities.tolist(),suspicion=dict(belief.by_player),
            memory_events=[asdict(e) for e in memory.events]))
    return capture
