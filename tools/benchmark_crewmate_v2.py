"""Bounded exact-tick engineering comparison. No learned policies or training."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from among_us_map import get_map
from navigation_service import NavigationPlanner
from phase3_runner import canonical_bytes
from social_deduction.phase3_api import IntentKind
from crewmate_v2.config import LobbySettings
from crewmate_v2.runtime import V2StrategicMatch
from crewmate_v2.provenance import sources

ENGINEERING_SEEDS=(2612000,2612001,2612002)


def run(seed,fast,deadline):
    started=perf_counter()
    match=V2StrategicMatch(seed,fast=fast,planner=NavigationPlanner(get_map(),cache_size=96))
    game=match.game;extra=hashlib.sha256();actions=Counter();body_observers=set();witnesses=set();no_action_windows=0
    while not game.is_terminal:
        if perf_counter()>deadline:raise TimeoutError('Bounded engineering benchmark deadline exceeded')
        submitted={}
        if game.tick%match.decision_ticks==0:
            observations={pid:game.observe_v2(pid) for pid in sorted(match.controllers)}
            game.record_observations({pid:o.actor for pid,o in observations.items()})
            for pid,o in observations.items():
                match._record_observation(pid,o.actor)
                if not o.actor.own.active:continue
                extra.update(canonical_bytes((pid,o.task_bar,o.events,o.own_tasks)))
                if o.actor.visible_bodies:body_observers.add(pid)
                if any(type(e.payload).__name__=='WitnessedElimination' for e in o.actor.evidence):witnesses.add(pid)
                if not o.actor.actions:
                    no_action_windows+=1;continue
                action=match.controllers[pid].decide(o.actor)
                match._actor_hash.update(canonical_bytes((pid,action)))
                match._actor_hash.update(b'\n');actions[action.kind.value]+=1
                if action.kind is not IntentKind.WAIT:submitted[pid]=action
        game.step(submitted)
    result=dict(seconds=perf_counter()-started,simulated_seconds=game.time,ticks=game.tick,
        illegal_actions=game.metrics['illegal_actions'],navigation_failures=game.metrics['navigation_failures'],
        ghost_crewmates=len(game.ghosts),task_completions=game.metrics['task_completions'],
        reports=game.metrics['reports'],votes=game.metrics['votes'],reason=game.result.reason,
        action_counts=dict(actions),body_observers=len(body_observers),witnesses=len(witnesses),
        no_action_actor_windows=no_action_windows,
        truth=match.truth_trace_hash,actor=match.actor_trace_hash,v2=extra.hexdigest())
    if result['illegal_actions'] or result['navigation_failures']:raise RuntimeError('Unsafe engineering execution')
    return result


def benchmark(output,repeats=2):
    if repeats not in (1,2):raise ValueError('Only bounded one/two repeats are authorized by this tool')
    output=Path(output)
    if output.exists():raise FileExistsError('Do not overwrite engineering evidence')
    deadline=perf_counter()+240.*repeats;pairs=[]
    for repeat in range(repeats):
        for index,seed in enumerate(ENGINEERING_SEEDS):
            order=(False,True) if (repeat+index)%2==0 else (True,False)
            result={}
            for fast in order:
                result[fast]=run(seed,fast,deadline)
                print(f"engineering seed={seed} repeat={repeat} fast={fast} seconds={result[fast]['seconds']:.3f}",flush=True)
            reference,fast=result[False],result[True]
            for key in reference:
                if key!='seconds' and reference[key]!=fast[key]:raise AssertionError(f'Behavior changed: {key}')
            pairs.append((reference,fast))
    # Repeat determinism is checked from semantic hashes, not wall-clock counters.
    if repeats==2:
        for i in range(len(ENGINEERING_SEEDS)):
            for side in (0,1):
                for key in ('truth','actor','v2'):
                    assert pairs[i][side][key]==pairs[i+len(ENGINEERING_SEEDS)][side][key]
    reference_seconds=[a['seconds'] for a,b in pairs];fast_seconds=[b['seconds'] for a,b in pairs]
    timing_start=len(ENGINEERING_SEEDS) if repeats==2 else 0
    primary_reference=reference_seconds[timing_start:];primary_fast=fast_seconds[timing_start:]
    document=dict(status='bounded_engineering_only_no_training',lobby=LobbySettings().to_dict(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scientific_sources_sha256=hashlib.sha256(canonical_bytes(sources())).hexdigest(),
        benchmark_driver_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        engineering_seeds=list(ENGINEERING_SEEDS),distinct_scenarios=len(ENGINEERING_SEEDS),
        repeats=repeats,executions=len(pairs)*2,physics_ticks_per_second=1/LobbySettings().dt,
        operational_ceiling_seconds=240*repeats,
        primary_timing=dict(policy='second repetition, excluding first-repetition warmup' if repeats==2 else 'single repetition',
            full_projection_median=statistics.median(primary_reference),lazy_projection_median=statistics.median(primary_fast),
            median_paired_speedup=statistics.median(a/b for a,b in zip(primary_reference,primary_fast)),
            total_speedup=sum(primary_reference)/sum(primary_fast),paired_scenarios=len(primary_reference)),
        full_projection_seconds=reference_seconds,lazy_projection_seconds=fast_seconds,
        full_projection_median=statistics.median(reference_seconds),lazy_projection_median=statistics.median(fast_seconds),
        median_paired_speedup=statistics.median(a/b for a,b in zip(reference_seconds,fast_seconds)),
        total_speedup=sum(reference_seconds)/sum(fast_seconds),
        exact_actor_action_event_result_equivalence=True,exact_v2_public_packet_equivalence=True,
        deterministic_repeat_equivalence=repeats==2,
        ticks_per_method=sum(a['ticks'] for a,b in pairs),simulated_seconds_per_method=sum(a['simulated_seconds'] for a,b in pairs),
        aggregate_per_method={key:sum(a[key] for a,b in pairs) for key in
            ('illegal_actions','navigation_failures','ghost_crewmates','task_completions','reports','votes','body_observers','witnesses','no_action_actor_windows')},
        outcomes_per_method=dict(Counter(a['reason'] for a,b in pairs)),python=platform.python_version(),
        notes=['No rendering or sleeping in either execution path.','Same physics dt, timers, observation and decision intervals.',
               'All scheduled actor inputs and submitted actions were hashed. No private replay or individual role dataset is published.',
               'Candidate event jumps/coarse ticks/time compression not adopted; see design limitations.'],
        final_test_opened=False,phase6_1_acceptance='pending')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(document,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,choices=(1,2),default=2)
    args=parser.parse_args();benchmark(args.output,args.repeats)
