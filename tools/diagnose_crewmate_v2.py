"""Bounded, development-only scripted opportunity diagnostic; never trains a model."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from random import Random
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from among_us_map import get_map
from navigation_service import NavigationPlanner
from phase3_runner import behavior_seed, canonical, canonical_bytes
from phase4_scripts import StudyController
from phase6.config import PARTITIONS
from crewmate_v2.config import LobbySettings, MatchMixture
from crewmate_v2.provenance import sources
from crewmate_v2.runtime import V2StrategicMatch
from social_deduction.actor import Phase, Role
from social_deduction.phase3_api import IntentKind, WitnessedElimination

BASE = 'e62ba0b6c33ffd0317f78e18b4289d778cd5af2b'
SEEDS = tuple(range(2630000, 2630008))
FIXTURE_SEEDS = tuple(range(2630100, 2630110))
BUDGET = 900.


class DiagnosticDeadline(TimeoutError):
    def __init__(self, game):
        super().__init__('900-second diagnostic budget exhausted; no incomplete match scored')
        self.partial = dict(seed=game.seed,tick=game.tick,time=game.time,events=canonical(game.event_log))


def seed_guard():
    for seed in SEEDS + FIXTURE_SEEDS:
        assert all(not low <= seed <= high for low, high in PARTITIONS.values())
        assert not 2610000 <= seed <= 2612999


class StaggeredReporter(StudyController):
    """Actor-packet-only probe. No world, evaluator or shared memory reference."""
    def __init__(self, seed, style):
        super().__init__(seed, style)
        self.report_delay = Random(seed ^ 0xD1A69057).choice((0., 18.))
        self.body_first_seen = {}

    def _observe(self, obs):
        super()._observe(obs)
        for body in obs.visible_bodies:
            self.body_first_seen.setdefault(body.victim_id, obs.time_seconds)

    def _report(self, obs):
        eligible = {bid for bid, when in self.body_first_seen.items()
                    if obs.time_seconds - when >= self.report_delay - 1e-9}
        # Keep the parent's action/geometry logic; never manufacture an action.
        action = super()._report(obs)
        return action if action is not None and action.target in eligible else None


class OpportunityMeter:
    """Observation counters; hidden role scoring is a separate post-game operation."""
    def __init__(self):
        self.counts = Counter()
        self.phase_counts = defaultdict(Counter)
        self.report_pairs = defaultdict(set)
        self.body_pairs = set()
        self.witness_pairs = set()
        self.evidence_ticks = {}
        self.vote_opportunities = defaultdict(set)
        self.vote_windows = Counter()
        self.submitted_reports = set()
        self.explicit_votes = {}
        self.first_kill_scheduled = None

    def observe(self, obs, action):
        if not obs.own.active:
            return
        role = obs.own.role.value
        phase = obs.context.phase.value
        kinds = {a.kind for a in obs.actions}
        c = self.phase_counts[phase]
        for target in (self.counts, c):
            target['living_windows'] += 1
            if not obs.actions:
                target['forced_no_actions'] += 1
            elif kinds <= {IntentKind.WAIT}:
                target['wait_only'] += 1
            else:
                target['actionable_windows'] += 1
                target['multiple_nonwait_choices'] += sum(a.kind is not IntentKind.WAIT for a in obs.actions) > 1
                if action is not None and action.kind is IntentKind.WAIT:
                    target['selected_wait_with_actions'] += 1
        if IntentKind.KILL in kinds and self.first_kill_scheduled is None:
            self.first_kill_scheduled = obs.time_seconds
        for body in obs.visible_bodies:
            if obs.own.role is Role.CREWMATE:
                pair = (obs.own.player_id, body.victim_id)
                self.body_pairs.add(pair)
                self.evidence_ticks.setdefault(pair, obs.tick)
        for e in obs.evidence:
            if (type(e.payload) is WitnessedElimination and obs.own.role is Role.CREWMATE
                    and e.payload.victim_id != obs.own.player_id):
                pair = (obs.own.player_id, e.payload.victim_id)
                self.witness_pairs.add(pair)
                self.evidence_ticks.setdefault(pair, obs.tick)
        reports = [a for a in obs.actions if a.kind is IntentKind.REPORT]
        if reports:
            self.counts[role + '_report_legal_windows'] += 1
            for a in reports:
                self.report_pairs[role].add((obs.own.player_id, a.target))
        if IntentKind.VOTE in kinds or IntentKind.SKIP in kinds:
            self.vote_opportunities[role].add((obs.context.meeting_id, obs.own.player_id))
            self.vote_windows[role] += 1
        if action is not None:
            self.counts['selected_' + action.kind.value] += 1
            if action.kind is IntentKind.REPORT:
                self.counts[role + '_report_submissions'] += 1
                self.submitted_reports.add((obs.tick, obs.own.player_id, action.target))
            if action.kind in (IntentKind.VOTE, IntentKind.SKIP):
                self.explicit_votes[(obs.context.meeting_id, obs.own.player_id)] = dict(
                    target=action.target if action.kind is IntentKind.VOTE else None,
                    witnessed=any(p[0] == obs.own.player_id for p in self.witness_pairs))


def offline_score(game, meter, first_kill_legal, phase_seconds, phase_wall, all_balloted_seconds, curves):
    """Privileged roles/truth are accessed here, after controller execution only."""
    roles = {pid: p.role.value for pid, p in game.players.items()}
    inactive = set()
    counts = Counter(meter.counts)
    reports = []
    meetings = []
    physical_witnesses = set()
    completed = []
    first_kill = None
    resolution_ids = set()
    voting_started = set()
    for e in game.event_log:
        kind = e['kind']
        if kind == 'elimination':
            inactive.add(e['victim_id'])
            counts['kills'] += 1
            if first_kill is None:
                first_kill = e['time']
        elif kind == 'elimination_witnessed':
            if roles[e['observer_id']] == 'crewmate' and e['observer_id'] != e['victim_id']:
                physical_witnesses.add((e['observer_id'], e['victim_id']))
        elif kind == 'ejection' and e['player_id'] is not None:
            inactive.add(e['player_id'])
        elif kind == 'task_completed':
            ghost = e['player_id'] in inactive
            counts['tasks_ghost' if ghost else 'tasks_living'] += 1
            completed.append((e['time'], ghost))
        elif kind == 'report':
            reports.append(e)
            counts[roles[e['reporter_id']] + '_reports'] += 1
        elif kind == 'meeting_started':
            meetings.append(e)
        elif kind == 'voting_started':
            voting_started.add(e['meeting_id'])
        elif kind == 'meeting_resolution':
            resolution_ids.add(e['meeting_id'])
        elif kind == 'claim':
            counts[roles[e['speaker_id']] + '_claims'] += 1
        elif kind == 'vote_submitted':
            role = roles[e['voter_id']]
            key = (e['meeting_id'], e['voter_id'])
            explicit = meter.explicit_votes.get(key)
            if e['target_id'] is None:
                counts[role + ('_explicit_skips' if explicit is not None else '_automatic_skips')] += 1
            elif role == 'crewmate':
                correct = roles[e['target_id']] == 'impostor'
                counts['crewmate_correct_votes' if correct else 'crewmate_incorrect_votes'] += 1
                if correct and explicit and explicit['witnessed']:
                    counts['crewmate_correct_votes_with_direct_witness'] += 1
            else:
                counts['impostor_non_skip_votes'] += 1
    accepted = {(e['tick'], e['reporter_id'], e['victim_id']) for e in reports}
    counts['report_submissions'] = len(meter.submitted_reports)
    counts['report_submissions_not_accepted'] = len(meter.submitted_reports - accepted)
    for tick,pid,victim in meter.submitted_reports - accepted:
        counts[roles[pid] + '_report_submissions_not_accepted'] += 1
    counts['crewmate_body_observer_pairs'] = len(meter.body_pairs)
    counts['bodies_seen_by_crew'] = len({p[1] for p in meter.body_pairs})
    counts['physical_surviving_crew_witness_pairs'] = len(physical_witnesses)
    counts['delivered_surviving_crew_witness_pairs'] = len(meter.witness_pairs)
    counts['meetings'] = len(meetings)
    counts['meetings_reaching_voting'] = len(voting_started)
    counts['meetings_resolved'] = len(resolution_ids)
    counts['task_end_during_discussion'] = int(game.result.reason == 'TASKS_COMPLETED' and game.phase is Phase.FINISHED
        and any(e['meeting_id'] not in voting_started for e in meetings))
    counts['task_end_before_meeting_result'] = int(game.result.reason == 'TASKS_COMPLETED'
        and any(e['meeting_id'] not in resolution_ids for e in meetings))
    # A report, evidence for that body held by a participating crew actor, and a
    # legal crew ballot. Does not equate exposure with accurate deduction.
    for meeting in meetings:
        prior_reports = [r for r in reports if r['tick'] == meeting['tick']]
        if not prior_reports:
            continue
        victim = prior_reports[-1]['victim_id']
        informed = any(pid in meeting['participants'] and body == victim and tick <= meeting['tick']
                       for (pid, body), tick in meter.evidence_ticks.items())
        if informed:
            counts['crew_informed_report_meetings'] += 1
            if any(mid == meeting['meeting_id'] for mid, pid in meter.vote_opportunities['crewmate']):
                counts['deduction_sequence_meetings'] += 1
    for role in ('crewmate', 'impostor'):
        counts[role + '_report_unique_pairs'] = len(meter.report_pairs[role])
        counts[role + '_vote_unique_opportunities'] = len(meter.vote_opportunities[role])
        counts[role + '_vote_legal_windows'] = meter.vote_windows[role]
    for row in curves:
        row['ghost_completed'] = sum(ghost for time, ghost in completed if time <= row['time'] + 1e-9)
    return dict(seed=game.seed, condition='B' if game.deduction_only else 'A',
        duration=game.time, winner=game.result.winner.value if game.result.winner else None,
        reason=game.result.reason, counts=dict(counts),
        first_kill_legal=first_kill_legal, first_kill_scheduled=meter.first_kill_scheduled, first_kill=first_kill,
        phase_simulated_seconds=dict(phase_seconds), phase_wall_seconds=dict(phase_wall),
        all_balloted_voting_seconds=all_balloted_seconds, phase_actor_windows={k:dict(v) for k,v in meter.phase_counts.items()},
        task_curve=curves, final_tasks_completed=sum(t.progress >= 1 for t in game.tasks.values()),
        total_tasks=game.initial_task_count,
        task_win_requires_ghost_progress_at_this_time=bool(game.result.reason == 'TASKS_COMPLETED' and counts['tasks_ghost']),
        illegal_actions=game.metrics['illegal_actions'], navigation_failures=game.metrics['navigation_failures'])


def run_game(seed, condition, profile, deadline, lobby=None):
    started = perf_counter()
    match = V2StrategicMatch(seed, lobby=lobby or LobbySettings(), mixture=MatchMixture(float(condition == 'B')),
                             planner=NavigationPlanner(get_map(), cache_size=96), fast=True)
    game = match.game
    if profile == 'staggered':
        match.controllers = {pid: StaggeredReporter(behavior_seed(seed,pid), controller.style)
                             for pid, controller in match.controllers.items()}
    elif profile != 'baseline':
        raise ValueError('Unknown reporting probe')
    meter = OpportunityMeter()
    extra_hash = hashlib.sha256()
    prefixes = {}
    phase_seconds, phase_wall = Counter(), Counter()
    all_balloted_seconds = 0.
    first_kill_legal = None
    curves = []
    while not game.is_terminal:
        if perf_counter() > deadline:
            raise DiagnosticDeadline(game)
        tick_started = perf_counter()
        phase = game.phase.value
        pre_tick = game.tick
        # Additional trusted measurement at physics rate; not an actor callback.
        if first_kill_legal is None and game.phase is Phase.ROAMING:
            for p in game.players.values():
                if p.role is Role.IMPOSTOR and p.active and any(a.kind is IntentKind.KILL for a in game.observe(p.player_id).actions):
                    first_kill_legal = game.time
        if game.phase is Phase.VOTING and set(game.participants) <= set(game.votes):
            all_balloted_seconds += game.config.dt
        submitted = {}
        if game.tick % match.decision_ticks == 0:
            envelopes = {pid:game.observe_v2(pid) for pid in sorted(match.controllers)}
            game.record_observations({pid:e.actor for pid,e in envelopes.items()})
            for pid, envelope in envelopes.items():
                obs = envelope.actor
                match._record_observation(pid, obs)
                extra_hash.update(canonical_bytes((pid,envelope.task_bar,envelope.events,envelope.own_tasks)))
                action = match.controllers[pid].decide(obs) if obs.own.active and obs.actions else None
                meter.observe(obs, action)
                if action is not None:
                    match._actor_hash.update(canonical_bytes((pid,action)))
                    if action.kind is not IntentKind.WAIT:
                        submitted[pid] = action
        game.step(submitted)
        if game.metrics['illegal_actions'] or game.metrics['navigation_failures']:
            raise RuntimeError('Major legality/navigation issue; diagnostic stopped')
        elapsed = (game.tick-pre_tick)*game.config.dt
        phase_seconds[phase] += elapsed
        prefixes[game.tick] = hashlib.sha256(canonical_bytes((match.actor_trace_hash,extra_hash.hexdigest()))).hexdigest()
        if game.tick % 150 == 0 or game.is_terminal:
            curves.append(dict(time=game.time,completed=sum(t.progress>=1 for t in game.tasks.values()),
                total=game.initial_task_count,work_fraction=sum(t.progress for t in game.tasks.values())/game.initial_task_count))
        phase_wall[phase] += perf_counter()-tick_started
    result = offline_score(game,meter,first_kill_legal,phase_seconds,phase_wall,all_balloted_seconds,curves)
    result.update(profile=profile, wall_seconds=perf_counter()-started, terminal_tick=game.tick,
                  actor_sha256=match.actor_trace_hash, public_extra_sha256=extra_hash.hexdigest(), truth_sha256=match.truth_trace_hash)
    return result, prefixes, match.replay_document()


def ratio(num, den):
    return num / den if den else None


def summarize(rows):
    counts = Counter()
    for row in rows:
        counts.update(row['counts'])
    phases = Counter()
    walls = Counter()
    for row in rows:
        phases.update(row['phase_simulated_seconds'])
        walls.update(row['phase_wall_seconds'])
    result = dict(matches=len(rows),win_reasons=dict(Counter(r['reason'] for r in rows)),
        winners=dict(Counter(str(r['winner']) for r in rows)),counts=dict(counts),
        median_duration=statistics.median(r['duration'] for r in rows) if rows else None,
        simulated_seconds=sum(r['duration'] for r in rows),wall_seconds=sum(r['wall_seconds'] for r in rows),
        phase_simulated_seconds=dict(phases),phase_wall_seconds=dict(walls),
        all_balloted_voting_seconds=sum(r['all_balloted_voting_seconds'] for r in rows),
        ghost_required_task_wins=sum(r['task_win_requires_ghost_progress_at_this_time'] for r in rows))
    result['ratios'] = dict(
        crew_reports_per_legal_actor_body_pair=ratio(counts['crewmate_reports'],counts['crewmate_report_unique_pairs']),
        impostor_reports_per_legal_actor_body_pair=ratio(counts['impostor_reports'],counts['impostor_report_unique_pairs']),
        crew_share_of_accepted_reports=ratio(counts['crewmate_reports'],counts['crewmate_reports']+counts['impostor_reports']),
        crew_cast_accuracy=ratio(counts['crewmate_correct_votes'],counts['crewmate_correct_votes']+counts['crewmate_incorrect_votes']),
        crew_non_skip_participation=ratio(counts['crewmate_correct_votes']+counts['crewmate_incorrect_votes'],counts['crewmate_vote_unique_opportunities']),
        forced_no_action_share=ratio(counts['forced_no_actions'],counts['living_windows']),
        ghost_completion_share=ratio(counts['tasks_ghost'],counts['tasks_ghost']+counts['tasks_living']),
        balloted_wait_share_of_voting=ratio(result['all_balloted_voting_seconds'],phases['voting']))
    result['first_event_times'] = {}
    for key in ('first_kill_legal','first_kill_scheduled','first_kill'):
        observed = [r[key] for r in rows if r[key] is not None]
        result['first_event_times'][key] = dict(observed=len(observed),censored=len(rows)-len(observed),
            median=statistics.median(observed) if observed else None,min=min(observed) if observed else None,max=max(observed) if observed else None)
    result['match_exposures'] = {key:sum(r['counts'].get(key,0)>0 for r in rows) for key in (
        'kills','physical_surviving_crew_witness_pairs','crewmate_body_observer_pairs',
        'crewmate_report_unique_pairs','crewmate_reports','meetings','meetings_reaching_voting',
        'meetings_resolved','deduction_sequence_meetings','task_end_before_meeting_result')}
    bins = sorted({r['time'] for row in rows for r in row['task_curve'] if abs(r['time']%30)<1e-8})
    result['task_curve'] = []
    for time in bins:
        points = [p for r in rows for p in r['task_curve'] if p['time']==time]
        result['task_curve'].append(dict(time=time,matches_still_observed=len(points),
            mean_completed=statistics.mean(p['completed'] for p in points),
            mean_work_fraction=statistics.mean(p['work_fraction'] for p in points),
            mean_ghost_completed=statistics.mean(p['ghost_completed'] for p in points)))
    return result


def write_json(path, record):
    Path(path).write_text(json.dumps(canonical(record),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def diagnostic(output, budget=BUDGET):
    if budget != BUDGET:
        raise ValueError('Registered budget is fixed at 900 wall seconds')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Never overwrite diagnostic evidence')
    seed_guard()
    output.mkdir(parents=True)
    initial_sources = sources()
    started = perf_counter()
    deadline = started+budget
    rows, completed_pairs, baseline_rows = [], [], []
    private_dir = output/'private'
    private_dir.mkdir()
    write_json(output/'registration.json',dict(base=BASE,seeds=SEEDS,budget_seconds=budget,
        sources=initial_sources,driver_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()))
    def execute(seed,condition,profile,label=''):
        row,prefix,replay = run_game(seed,condition,profile,deadline)
        key=f'{profile}-{seed}-{condition}{label}'
        write_json(private_dir/(key+'.json'),dict(metrics=row,prefix_hashes=prefix,replay=replay))
        print(f'{key}: {row["reason"]} sim={row["duration"]:.1f}s wall={row["wall_seconds"]:.2f}s',flush=True)
        return row,prefix
    status='complete'
    replay_ok=False
    try:
        for index,seed in enumerate(SEEDS):
            pair={}
            for condition in (('A','B') if index%2==0 else ('B','A')):
                row,prefix=execute(seed,condition,'baseline')
                pair[condition]=(row,prefix)
                rows.append(row);baseline_rows.append(row)
                write_json(private_dir/'completed-matches.json',rows)
            a,ap=pair['A'];b,bp=pair['B']
            assert a['terminal_tick']<=b['terminal_tick']
            assert all(ap[t]==bp[t] for t in ap),'Mode marker or paired replay divergence'
            completed_pairs.append(dict(seed=seed,profile='baseline',prefix_equal=True,
                extra_seconds=b['duration']-a['duration'],a_reason=a['reason'],b_reason=b['reason']))
            if index==0:
                repeat,_=execute(seed,'A','baseline','-repeat')
                ignore={'wall_seconds','phase_wall_seconds'}
                assert {k:v for k,v in repeat.items() if k not in ignore}=={k:v for k,v in a.items() if k not in ignore}
                replay_ok=True
        reports=Counter()
        for r in baseline_rows:
            for role in ('crewmate','impostor'):
                reports[role]+=r['counts'].get(role+'_reports',0)
        total=sum(reports.values())
        if not total or max(reports.values())/total>=.9:
            pair_costs=[sum(r['wall_seconds'] for r in baseline_rows if r['seed']==seed) for seed in SEEDS]
            for seed in SEEDS[:2]:
                if deadline-perf_counter()<max(pair_costs)*1.4+5:
                    break
                pair={}
                for condition in ('A','B'):
                    row,prefix=execute(seed,condition,'staggered')
                    rows.append(row);pair[condition]=(row,prefix)
                    write_json(private_dir/'completed-matches.json',rows)
                a,ap=pair['A'];b,bp=pair['B']
                assert all(ap[t]==bp[t] for t in ap)
                completed_pairs.append(dict(seed=seed,profile='staggered',prefix_equal=True,
                    extra_seconds=b['duration']-a['duration'],a_reason=a['reason'],b_reason=b['reason']))
    except TimeoutError as error:
        status='budget_stopped'
        if isinstance(error,DiagnosticDeadline):
            write_json(private_dir/'incomplete.json',error.partial)
        write_json(output/'stop.json',dict(status=status,reason=str(error),completed=len(rows)))
    finally:
        assert sources()==initial_sources,'Source drift during diagnostic'
    groups={profile+'_'+condition:summarize([r for r in rows if r['profile']==profile and r['condition']==condition])
        for profile in ('baseline','staggered') for condition in ('A','B') if any(r['profile']==profile for r in rows)}
    record=dict(status=status,development_only=True,phase6_1_acceptance='pending',base_commit=BASE,
        diagnostic_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        canonical_sources_sha256=hashlib.sha256(canonical_bytes(initial_sources)).hexdigest(),
        lobby=LobbySettings().to_dict(),seeds=list(SEEDS),registered_primary_scenarios=len(SEEDS),
        distinct_primary_scenarios=len({r['seed'] for r in baseline_rows}),
        completed_executions=len(rows),replay_executions=int(replay_ok),deterministic_replay_verified=replay_ok,
        paired_prefixes=completed_pairs,groups=groups,wall_seconds=perf_counter()-started,budget_seconds=budget,
        final_test_opened=False,models_loaded=False,training=False,
        limits=['Scripted policies, not learned agents or eight independent populations.',
                'Eight paired primary scenarios; conditional reporter probe only two repeated scenarios.',
                'Report propensities compete within matches; actor-windows are not independent trials.',
                'Ghost task attribution is descriptive, not a causal no-ghost ablation.',
                'Existing scripts consume legacy actor packets, not new V2 visual/confirmed-role channels.'])
    write_json(output/'aggregate.json',record)
    print(f'DONE status={status} wall={record["wall_seconds"]:.2f}s executions={len(rows)} + replay={int(replay_ok)}',flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    diagnostic(args.output)
