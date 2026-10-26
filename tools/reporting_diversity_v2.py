"""Observation-only reporting perturbations and offline shortcut assay. No training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from random import Random
import statistics
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from among_us_map import get_map
from navigation_service import NavigationPlanner
from phase3_runner import behavior_seed,canonical_bytes
from phase4_scripts import StudyController
from crewmate_v2.config import LobbySettings,MatchMixture
from crewmate_v2.provenance import sources
from crewmate_v2.runtime import V2StrategicMatch
from social_deduction.actor import BodySeen,Meeting,Phase,Report,Role
from social_deduction.phase3_api import ClaimKind,IntentKind,StructuredClaim,WitnessedElimination
from tools.diagnose_crewmate_v2 import (
    DiagnosticDeadline,OpportunityMeter,SEEDS,offline_score,ratio,seed_guard,summarize,write_json,
)

BASE='cad6c34'
STYLES=('prompt','inspect','bypass')
BUDGET=900.


class DiverseReporter(StudyController):
    """Explicit diagnostic scripts only; never installed in a trainable policy."""
    def __init__(self,seed,style):
        super().__init__(seed,style)
        self.reporting_style=Random(seed^0xD17E2517).choice(STYLES)
        self.first_body_sight={}

    def _observe(self,obs):
        super()._observe(obs)
        for body in obs.visible_bodies:
            self.first_body_sight.setdefault(body.victim_id,obs.time_seconds)

    def _report(self,obs):
        if self.reporting_style=='bypass':return None
        action=super()._report(obs)
        if action is None or self.reporting_style=='prompt' or action.kind is not IntentKind.REPORT:
            return action
        if obs.time_seconds-self.first_body_sight[action.target]>=6.-1e-9:return action
        # Stay at an observed body, without manufacturing a motion intention.
        for kind in (IntentKind.CANCEL_NAVIGATION,IntentKind.WAIT):
            if kind is IntentKind.CANCEL_NAVIGATION and obs.navigation_status!='moving' and obs.interaction_task is None:
                continue
            choices=self._available(obs,kind)
            if choices:return choices[0]
        return None


def report_context(obs):
    """Only authenticated public events and public participants, no role lookup."""
    meetings=[e for e in obs.evidence if type(e.payload) is Meeting and e.payload.meeting_id==obs.context.meeting_id]
    if not meetings:return None,None
    tick=meetings[-1].tick
    reports=[e.payload for e in obs.evidence if type(e.payload) is Report and e.tick==tick]
    return (reports[-1].reporter_id,reports[-1].victim_id) if reports else (None,None)


def reporter_weights(candidates,reporter):
    """Predictions from public candidate IDs and current reporter identity only."""
    candidates=tuple(candidates)
    if not candidates:return dict(uniform={},accuse={},avoid={})
    uniform={pid:1./len(candidates) for pid in candidates}
    accuse={pid:float(pid==reporter) for pid in candidates} if reporter in candidates else dict(uniform)
    remaining=[pid for pid in candidates if pid!=reporter]
    avoid={pid:(1./len(remaining) if pid in remaining else 0.) for pid in candidates} if remaining else dict(uniform)
    return dict(uniform=uniform,accuse=accuse,avoid=avoid)


class ExposureTracker:
    """Personal packet measurements. Attribution to hidden roles happens later."""
    def __init__(self):
        self.pairs={};self.legal_actor_windows=Counter();self.ballots={}

    def observe(self,obs,action):
        if not obs.own.active:return
        pid=obs.own.player_id
        for body in obs.visible_bodies:
            row=self.pairs.setdefault((pid,body.victim_id),dict(actor=pid,body=body.victim_id,
                first_visible=obs.time_seconds,last_visible=obs.time_seconds,visible_windows=0,
                first_legal=None,last_legal=None,legal_windows=0,submissions=[]))
            row['last_visible']=obs.time_seconds;row['visible_windows']+=1
        legal=[a for a in obs.actions if a.kind is IntentKind.REPORT]
        if legal:self.legal_actor_windows[pid]+=1
        for candidate in legal:
            row=self.pairs[(pid,candidate.target)]
            if row['first_legal'] is None:row['first_legal']=obs.time_seconds
            row['last_legal']=obs.time_seconds;row['legal_windows']+=1
        if action is not None and action.kind is IntentKind.REPORT:
            assert action in obs.actions,'Reporter manufactured a nonlegal action'
            self.pairs[(pid,action.target)]['submissions'].append(dict(tick=obs.tick,time=obs.time_seconds))
        if any(a.kind in (IntentKind.VOTE,IntentKind.SKIP) for a in obs.actions):
            key=(obs.context.meeting_id,pid)
            if key in self.ballots:return
            reporter,body=report_context(obs)
            candidates=tuple(p for p in obs.context.participants if p!=pid)
            direct=any(type(e.payload) is WitnessedElimination and e.payload.killer_id in candidates
                       and e.payload.victim_id!=pid for e in obs.evidence)
            body_info=(pid,body) in self.pairs
            heard=any(type(e.payload) is StructuredClaim and e.payload.speaker_id!=pid
                and e.payload.subject_id in candidates and e.payload.kind in
                (ClaimKind.SAW_ELIMINATION,ClaimKind.SAW_PLAYER_NEAR_BODY) for e in obs.evidence)
            self.ballots[key]=dict(actor=pid,meeting=key[0],time=obs.time_seconds,
                reporter=reporter,candidates=candidates,weights=reporter_weights(candidates,reporter),
                direct_witness=direct,body_info=body_info,heard_claim=heard)


def distribution(values):
    return dict(n=len(values),median=statistics.median(values) if values else None,
                min=min(values) if values else None,max=max(values) if values else None)


def roles_from(events):
    return {e['player_id']:e['role'] for e in events if e['kind']=='spawn'}


def score_ballots(ballots,roles,seed,condition,profile):
    scored=[]
    for b in ballots:
        if roles[b['actor']]!='crewmate':continue
        assert sum(roles[p]=='impostor' for p in b['candidates'])==1
        scored.append(dict(seed=seed,condition=condition,profile=profile,actor=b['actor'],meeting=b['meeting'],
            time=b['time'],direct_witness=b['direct_witness'],body_info=b['body_info'],heard_claim=b['heard_claim'],
            reporter_candidate=b['reporter'] in b['candidates'],
            scores={name:sum(w for p,w in weights.items() if roles[p]=='impostor') for name,weights in b['weights'].items()}))
    return scored


def finish_exposures(game,tracker,full_bar,profile):
    events=game.event_log;roles=roles_from(events)
    accepted={(e['reporter_id'],e['victim_id']):e for e in events if e['kind']=='report'}
    report_by_body={e['victim_id']:e for e in events if e['kind']=='report'}
    absent={e['victim_id']:e['time'] for e in events if e['kind']=='elimination'}
    absent.update({e['player_id']:e['time'] for e in events if e['kind']=='ejection' and e['player_id'] is not None})
    rows=[]
    for key,original in tracker.pairs.items():
        row=dict(original,role=roles[key[0]],accepted_time=accepted.get(key,{}).get('time'))
        end=report_by_body.get(key[1])
        endpoints=[(game.time,3,'match_'+game.result.reason)]
        if end:endpoints.append((end['time'],0,'accepted' if key in accepted else 'other_report'))
        if key[0] in absent:endpoints.append((absent[key[0]],1,'actor_absent'))
        endpoints.extend((e['time'],2,'other_meeting') for e in events if e['kind']=='meeting_started'
                         and e['time']>=row['first_visible'])
        row['censor_time'],_,row['censor_reason']=min(endpoints)
        row['unaccepted_submissions']=sum(not(key in accepted and s['tick']==accepted[key]['tick']) for s in row['submissions'])
        rows.append(row)
    group={}
    for role in ('crewmate','impostor'):
        selected=[r for r in rows if r['role']==role]
        legal=[r for r in selected if r['first_legal'] is not None]
        submitted=[r for r in legal if r['submissions']]
        own=[r for r in legal if r['accepted_time'] is not None]
        group[role]=dict(visible_unique=len(selected),visible_body_windows=sum(r['visible_windows'] for r in selected),
            legal_unique=len(legal),legal_body_windows=sum(r['legal_windows'] for r in legal),
            legal_actor_windows=sum(n for pid,n in tracker.legal_actor_windows.items() if roles[pid]==role),
            submitting_unique=len(submitted),submissions=sum(len(r['submissions']) for r in submitted),accepted=len(own),
            unaccepted_submissions=sum(r['unaccepted_submissions'] for r in legal),
            visible_never_legal=len(selected)-len(legal),legal_never_submitted=len(legal)-len(submitted),
            first_visible_before_full_bar=sum(full_bar is None or r['first_visible']<full_bar for r in selected),
            first_legal_before_full_bar=sum(full_bar is None or r['first_legal']<full_bar for r in legal),
            visible_to_legal=[r['first_legal']-r['first_visible'] for r in legal],
            legal_to_submit=[r['submissions'][0]['time']-r['first_legal'] for r in submitted],
            visible_to_accept=[r['accepted_time']-r['first_visible'] for r in own],
            legal_to_accept=[r['accepted_time']-r['first_legal'] for r in own],
            censored_legal_followup=[r['censor_time']-r['first_legal'] for r in legal if r['accepted_time'] is None],
            censor_reasons=dict(Counter(r['censor_reason'] for r in selected)))
    condition='B' if game.deduction_only else 'A'
    ballots=score_ballots(tracker.ballots.values(),roles,game.seed,condition,profile)
    for ballot in ballots:ballot['before_full_bar']=full_bar is None or ballot['time']<full_bar
    return dict(by_role=group,first_full_bar=full_bar,ballots=ballots,pairs=rows)


def run_game(seed,condition,profile,deadline,lobby=None):
    started=perf_counter()
    match=V2StrategicMatch(seed,lobby=lobby or LobbySettings(),mixture=MatchMixture(float(condition=='B')),
        planner=NavigationPlanner(get_map(),cache_size=96),fast=True)
    game=match.game
    if profile=='diverse':
        match.controllers={pid:DiverseReporter(behavior_seed(seed,pid),c.style) for pid,c in match.controllers.items()}
    elif profile!='baseline':raise ValueError('Unknown diagnostic profile')
    # The trusted meter queries the actor's OWN role/action packet, never passes it to another actor.
    kill_actor=next(pid for pid in match.controllers if game.observe(pid).own.role is Role.IMPOSTOR)
    meter=OpportunityMeter();tracker=ExposureTracker();extra=hashlib.sha256();prefixes={}
    phase_seconds=Counter();phase_wall=Counter();balloted=0.;first_legal=None;curves=[];full_bar=None
    while not game.is_terminal:
        if perf_counter()>deadline:raise DiagnosticDeadline(game)
        tick_started=perf_counter();phase=game.phase.value;pre_tick=game.tick
        if first_legal is None and game.phase is Phase.ROAMING:
            if any(a.kind is IntentKind.KILL for a in game.observe(kill_actor).actions):first_legal=game.time
        if game.phase is Phase.VOTING and set(game.participants)<=set(game.votes):balloted+=game.config.dt
        actions={}
        if game.tick%match.decision_ticks==0:
            envelopes={pid:game.observe_v2(pid) for pid in sorted(match.controllers)}
            game.record_observations({pid:e.actor for pid,e in envelopes.items()})
            for pid,envelope in envelopes.items():
                obs=envelope.actor;match._record_observation(pid,obs)
                extra.update(canonical_bytes((pid,envelope.task_bar,envelope.events,envelope.own_tasks)))
                action=match.controllers[pid].decide(obs) if obs.own.active and obs.actions else None
                meter.observe(obs,action);tracker.observe(obs,action)
                if action is not None:
                    match._actor_hash.update(canonical_bytes((pid,action)))
                    if action.kind is not IntentKind.WAIT:actions[pid]=action
        game.step(actions)
        if game.metrics['illegal_actions'] or game.metrics['navigation_failures']:raise RuntimeError('Legality/navigation blocker')
        phase_seconds[phase]+=(game.tick-pre_tick)*game.config.dt
        prefixes[game.tick]=hashlib.sha256(canonical_bytes((match.actor_trace_hash,extra.hexdigest()))).hexdigest()
        complete=sum(t.progress>=1 for t in game.tasks.values())
        if complete==game.initial_task_count and full_bar is None:full_bar=game.time
        if game.tick%150==0 or game.is_terminal:
            curves.append(dict(time=game.time,completed=complete,total=game.initial_task_count,
                work_fraction=sum(t.progress for t in game.tasks.values())/game.initial_task_count))
        phase_wall[phase]+=perf_counter()-tick_started
    row=offline_score(game,meter,first_legal,phase_seconds,phase_wall,balloted,curves)
    row.update(profile=profile,wall_seconds=perf_counter()-started,terminal_tick=game.tick,
        actor_sha256=match.actor_trace_hash,public_extra_sha256=extra.hexdigest(),truth_sha256=match.truth_trace_hash)
    exposed=finish_exposures(game,tracker,full_bar,profile)
    styles={pid:c.reporting_style for pid,c in match.controllers.items()} if profile=='diverse' else {}
    return row,prefixes,match.replay_document(),exposed,styles


def baseline_audit(path):
    """Score archived public events; do not generate any new baseline games."""
    doc=json.loads(Path(path).read_text());row=doc['metrics'];events=doc['replay']['events'];roles=roles_from(events)
    first={};reporter_for={};meeting_for={};ballots=[]
    for event in events:
        kind=event['kind']
        if kind=='body_observed':first.setdefault((event['observer_id'],event['victim_id']),event['time'])
        elif kind=='report':reporter_for[event['tick']]=event
        elif kind=='meeting_started':meeting_for[event['meeting_id']]=event
        elif kind=='voting_started':
            meeting=meeting_for[event['meeting_id']];report=reporter_for.get(meeting['tick'],{})
            reporter=report.get('reporter_id');victim=report.get('victim_id')
            for pid in meeting['participants']:
                candidates=[p for p in meeting['participants'] if p!=pid]
                direct=any(e['kind']=='elimination_witnessed' and e['observer_id']==pid and e['victim_id']!=pid
                    and e['killer_id'] in candidates and e['time']<=event['time'] for e in events)
                body_info=(pid,victim) in first and first[(pid,victim)]<=event['time']
                heard=any(e['kind']=='claim' and e['speaker_id']!=pid and e.get('subject_id') in candidates
                    and e.get('claim_kind') in ('saw_elimination','saw_player_near_body')
                    and e['time']<=event['time'] for e in events)
                ballots.append(dict(actor=pid,meeting=event['meeting_id'],time=event['time'],reporter=reporter,
                    candidates=candidates,weights=reporter_weights(candidates,reporter),
                    direct_witness=direct,body_info=body_info,heard_claim=heard))
    scored=score_ballots(ballots,roles,row['seed'],row['condition'],'baseline')
    assert len(scored)==row['counts']['crewmate_vote_unique_opportunities']
    by_role={}
    for role in ('crewmate','impostor'):
        c=row['counts'];visible=[key for key in first if roles[key[0]]==role]
        legal=c.get(role+'_report_unique_pairs',0);windows=c.get(role+'_report_legal_windows',0)
        submissions=c.get(role+'_report_submissions',0)
        assert legal==windows==submissions,'Archived data do not support one-window, zero-latency inference'
        accepted=[e for e in events if e['kind']=='report' and roles[e['reporter_id']]==role]
        by_role[role]=dict(visible_unique=len(visible),legal_unique=legal,legal_actor_windows=windows,
            submitting_unique=legal,submissions=submissions,accepted=len(accepted),
            unaccepted_submissions=c.get(role+'_report_submissions_not_accepted',0),
            visible_never_legal=len(visible)-legal,legal_never_submitted=0,
            legal_to_submit=[0.]*submissions,
            visible_to_accept=[e['time']-first[(e['reporter_id'],e['victim_id'])] for e in accepted],
            legal_to_accept=[0.]*len(accepted),
            latency_note='Scheduled legal-to-submit inferred from equality of legal windows and submissions; fine-grained baseline visibility/first-range records unavailable.')
    return row,dict(by_role=by_role,ballots=scored,first_full_bar=row['duration'] if row['reason']=='TASKS_COMPLETED' else None)


def heuristic_summary(records):
    if not records:return dict(n=0)
    seeds=sorted({b['seed'] for b in records})
    event={name:statistics.mean(b['scores'][name] for b in records) for name in ('uniform','accuse','avoid')}
    balanced={name:statistics.mean(statistics.mean(b['scores'][name] for b in records if b['seed']==seed)
        for seed in seeds) for name in event}
    return dict(n=len(records),distinct_scenarios=len(seeds),event_weighted=event,scenario_balanced=balanced,
        direct_witness_ballots=sum(b['direct_witness'] for b in records),body_info_ballots=sum(b['body_info'] for b in records),
        specific_hearsay_ballots=sum(b['heard_claim'] for b in records),
        direct_crime_exposure_ballots=sum(b['direct_witness'] or b['body_info'] for b in records))


def loso_summary(records):
    seeds=sorted({b['seed'] for b in records});evaluations=[];directions={}
    for seed in seeds:
        training=[b for b in records if b['seed']!=seed]
        if not training:continue
        score=heuristic_summary(training)['scenario_balanced']
        choice='accuse' if score['accuse']>=score['avoid'] else 'avoid'
        directions[str(seed)]=choice
        evaluations.extend(dict(seed=seed,score=b['scores'][choice],uniform=b['scores']['uniform']) for b in records if b['seed']==seed)
    return dict(folds=len(directions),held_out_direction=directions,
        event_weighted=statistics.mean(b['score'] for b in evaluations) if evaluations else None,
        scenario_balanced=statistics.mean(statistics.mean(b['score'] for b in evaluations if b['seed']==seed)
            for seed in seeds) if evaluations else None,
        uniform_event_weighted=statistics.mean(b['uniform'] for b in evaluations) if evaluations else None)


def exposure_summary(exposures):
    result={}
    for role in ('crewmate','impostor'):
        entries=[x['by_role'][role] for x in exposures];counts=Counter();samples=defaultdict(list)
        for entry in entries:
            for key,value in entry.items():
                if type(value) is int:counts[key]+=value
                elif type(value) is list:samples[key].extend(value)
        result[role]=dict(counts=dict(counts),latencies={k:distribution(v) for k,v in samples.items()},
            rates=dict(legal_given_visible=ratio(counts['legal_unique'],counts['visible_unique']),
                submitting_given_legal_pair=ratio(counts['submitting_unique'],counts['legal_unique']),
                submissions_per_legal_actor_window=ratio(counts['submissions'],counts['legal_actor_windows']),
                accepted_given_submission=ratio(counts['accepted'],counts['submissions'])))
    reports=sum(result[r]['counts']['accepted'] for r in result)
    result['reporter_impostor_precision']=ratio(result['impostor']['counts']['accepted'],reports)
    return result


def run_diagnostic(previous,output):
    started=perf_counter();deadline=started+BUDGET;previous=Path(previous);output=Path(output)
    if output.exists():raise FileExistsError('Never overwrite diagnostic evidence')
    seed_guard();output.mkdir(parents=True);private=output/'private';private.mkdir()
    old_hashes={p.relative_to(previous).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in previous.rglob('*') if p.is_file()}
    known=json.loads((previous/'private-file-hashes.json').read_text())
    assert all(old_hashes[p]==h for p,h in known.items())
    initial=sources();baselines=[];baseline_ex=[];rows=[];exposures=[];pairs=[];verified=False;replayed=False
    for seed in SEEDS:
        for condition in ('A','B'):
            row,ex=baseline_audit(previous/'private'/f'baseline-{seed}-{condition}.json')
            baselines.append(row);baseline_ex.append(ex)
    write_json(output/'registration.json',dict(base=BASE,seeds=SEEDS,budget_seconds=BUDGET,
        previous_hashes=old_hashes,canonical_sources=initial,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()))
    write_json(private/'baseline-audit.json',dict(rows=baselines,exposures=baseline_ex))
    def execute(seed,condition,profile,suffix=''):
        row,prefix,replay,ex,styles=run_game(seed,condition,profile,deadline)
        write_json(private/f'{profile}-{seed}-{condition}{suffix}.json',dict(metrics=row,prefix_hashes=prefix,
            replay=replay,exposures=ex,styles=styles))
        print(f'{profile}-{seed}-{condition}{suffix}: {row["reason"]} sim={row["duration"]:.1f}s wall={row["wall_seconds"]:.2f}s crew reports={row["counts"].get("crewmate_reports",0)}',flush=True)
        return row,prefix,ex
    status='complete';failure=None
    try:
        rerun,_,_=execute(SEEDS[0],'B','baseline','-verification')
        original=next(r for r in baselines if r['seed']==SEEDS[0] and r['condition']=='B')
        ignore={'wall_seconds','phase_wall_seconds'}
        assert {k:v for k,v in rerun.items() if k not in ignore}=={k:v for k,v in original.items() if k not in ignore},'Archived baseline reproduction failed'
        verified=True
        for index,seed in enumerate(SEEDS):
            pair={}
            for condition in (('A','B') if index%2==0 else ('B','A')):
                row,prefix,ex=execute(seed,condition,'diverse')
                rows.append(row);exposures.append(ex);pair[condition]=(row,prefix)
            a,ap=pair['A'];b,bp=pair['B']
            assert a['terminal_tick']<=b['terminal_tick'] and all(ap[t]==bp[t] for t in ap),'Match mode altered pre-terminal actor inputs'
            pairs.append(dict(seed=seed,prefix_equal=True,ordinary_reason=a['reason'],deduction_reason=b['reason']))
            if index==0:
                repeat,_,_=execute(seed,'A','diverse','-replay')
                assert {k:v for k,v in repeat.items() if k not in ignore}=={k:v for k,v in a.items() if k not in ignore},'Diverse replay failed'
                replayed=True
    except DiagnosticDeadline as error:
        status='budget_stopped';write_json(private/'incomplete.json',error.partial)
    except (AssertionError,RuntimeError) as error:
        status='blocked';failure=str(error)
    finally:
        assert sources()==initial,'Canonical source drift'
        assert all(hashlib.sha256((previous/p).read_bytes()).hexdigest()==h for p,h in old_hashes.items()),'Previous evidence changed'
    groups={};ballots_by_profile={}
    for profile,records,exs in (('baseline',baselines,baseline_ex),('diverse',rows,exposures)):
        ballots_by_profile[profile]=[b for ex in exs for b in ex['ballots']]
        for condition in ('A','B'):
            selected=[(r,e) for r,e in zip(records,exs) if r['condition']==condition]
            rr=[r for r,e in selected];ee=[e for r,e in selected]
            groups[profile+'_'+condition]=dict(game=summarize(rr),reporting=exposure_summary(ee),
                reporter_heuristics=heuristic_summary([b for e in ee for b in e['ballots']]))
    result=dict(status=status,failure=failure,development_only=True,phase6_1_acceptance='pending',
        base_commit=BASE,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        canonical_sources_sha256=hashlib.sha256(canonical_bytes(initial)).hexdigest(),lobby=LobbySettings().to_dict(),
        seeds=SEEDS,old_baseline_executions_reused=16,new_primary_executions=len(rows),
        baseline_verification_executions=int(verified),diverse_replay_executions=int(replayed),
        matched_mode_pairs=pairs,groups=groups,
        grouped_shortcut_loso={p:loso_summary(b) for p,b in ballots_by_profile.items()},
        wall_seconds=perf_counter()-started,budget_seconds=BUDGET,old_files_preserved=len(old_hashes),
        final_test_opened=False,neural_training=False,models_changed=False,
        limits=['Archived baseline has no full per-pair range/visibility timing records; zero legal-to-submit delay is inferred from counters.',
            'Eight reused development scenarios, not fresh generalization or independent paired conditions.',
            'Fixed role-independent styles can still correlate with roles by chance and behavior.',
            'No evidence that any new neural model learned deduction; scripts ignore added V2 visual/confirm channels.'])
    write_json(output/'aggregate.json',result)
    raw_hashes={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in private.glob('*.json')}
    write_json(output/'private-file-hashes.json',raw_hashes)
    print(f'DONE {status} new primary={len(rows)} wall={result["wall_seconds"]:.2f}s',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run_diagnostic(args.previous,args.output)
