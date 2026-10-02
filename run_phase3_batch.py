"""Seeded, window-free acceptance runs: python run_phase3_batch.py --matches 1000."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
import traceback


_PLANNER = None


def _initialize_worker():
    global _PLANNER
    from among_us_map import get_map
    from navigation_service import NavigationPlanner
    _PLANNER = NavigationPlanner(get_map(), cache_size=96)


def run_case(seed, config=None, replay=True, log_directory=None):
    """Return evidence for one primary match; replay is independently reset."""
    from phase3_engine import GameConfig
    from phase3_runner import ScriptedMatch
    if _PLANNER is None:
        _initialize_worker()
    selected = GameConfig(**config) if isinstance(config, dict) else config or GameConfig()
    row = {"seed": seed, "completed": 0, "crashes": 0, "invalid_states": 0,
           "timeouts": 0, "deterministic_replay_failures": 0, "replay_checked": 0,
           "winner": "", "reason": "", "error": ""}
    started = perf_counter()
    match = None
    try:
        match = ScriptedMatch(seed=seed, config=selected, planner=_PLANNER)
        result = match.run()
        row.update({
            "completed": int(match.game.is_terminal),
            "winner": result.winner.value if result.winner is not None else "draw",
            "reason": result.reason, "simulation_seconds": result.simulation_time,
            "timeouts": int("timeout" in result.reason.lower()),
            "truth_trace_hash": match.truth_trace_hash,
            "actor_trace_hash": match.actor_trace_hash,
            "trajectory_hash": match.trajectory_hash,
            "decision_calls": match.decision_calls,
            "observation_hash_seconds": match.observation_hash_seconds,
            "policy_seconds": match.policy_seconds,
            "event_count": len(match.game.event_log),
            **match.game.metrics,
        })
        row["runtime_seconds"] = perf_counter() - started
        if log_directory is not None:
            match.write_replay(Path(log_directory) / f"seed_{seed}.jsonl")
        if replay:
            replay_started = perf_counter()
            repeated = ScriptedMatch(seed=seed, config=selected, planner=_PLANNER)
            repeated.run()
            row["replay_checked"] = 1
            row["deterministic_replay_failures"] = int(
                repeated.trajectory_hash != row["trajectory_hash"])
            row["replay_runtime_seconds"] = perf_counter() - replay_started
            if row["deterministic_replay_failures"] and log_directory is not None:
                repeated.write_replay(Path(log_directory) / f"seed_{seed}_replay_mismatch.jsonl")
    except Exception as exc:
        row["crashes"] = 1
        row["invalid_states"] = int(isinstance(exc, AssertionError))
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
        row.setdefault("runtime_seconds", perf_counter() - started)
        if match is not None and log_directory is not None:
            match.write_replay(Path(log_directory) / f"seed_{seed}_failure.jsonl")
    return row


def _run_job(job):
    return run_case(*job)


def summarize(rows, config, wall_seconds):
    totals = {key: sum(row.get(key, 0) for row in rows) for key in (
        "completed", "crashes", "invalid_states", "timeouts", "replay_checked",
        "deterministic_replay_failures", "eliminations", "reports", "meetings", "votes",
        "task_completions", "illegal_actions", "navigation_failures", "steps", "event_count",
    )}
    winners = Counter(row["winner"] for row in rows if row["completed"])
    reasons = Counter(row["reason"] for row in rows if row["completed"])
    good = [row for row in rows if row["completed"]]
    means = {key: statistics.fmean(row.get(key, 0) for row in good) if good else 0.0
             for key in ("runtime_seconds", "simulation_seconds", "steps", "visibility_seconds",
                         "navigation_seconds", "physics_seconds", "event_log_seconds",
                         "observation_hash_seconds", "policy_seconds")}
    gate = bool(rows) and totals["completed"] == len(rows) and all(totals[key] == 0 for key in (
        "crashes", "invalid_states", "timeouts", "deterministic_replay_failures",
        "illegal_actions", "navigation_failures",
    )) and totals["replay_checked"] == len(rows)
    return {
        "schema": "phase3-batch-v1", "matches": len(rows), "totals": totals,
        "winners": dict(sorted(winners.items())), "win_reasons": dict(sorted(reasons.items())),
        "means": means, "configuration": config, "batch_wall_seconds": wall_seconds,
        "acceptance": "PASS" if gate and len(rows) >= 1000 else
                      "SUBSET_ONLY" if gate else "FAIL_OR_UNCHECKED",
        "notes": [
            "Primary matches count once; deterministic replay repeats do not inflate game totals.",
            "Trace comparison includes every full actor decision input/action and privileged event log.",
            "Information-boundary metamorphic tests are a separate required regression gate.",
            "A timeout is a draw/failure category, never a crew or impostor win.",
            "Wall timings include validation, actor hashing and concurrent machine activity.",
        ],
    }


def _provenance():
    root = Path(__file__).resolve().parent
    paths = ["phase3_engine.py", "phase3_bots.py", "phase3_runner.py", "run_phase3_batch.py",
             "social_deduction/phase3_api.py", "social_deduction/phase3_observation.py",
             "social_deduction/actor.py", "social_deduction/truth.py", "phase3_cli.py",
             "navigation_service.py", "among_us_map.py", "among_us_map_simulation.py",
             "assets/skeld/among_us_map.json"]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    return {"base_git_commit": commit, "python": sys.version, "platform": platform.platform(),
            "source_sha256": {path: sha256((root / path).read_bytes()).hexdigest()
                              for path in paths if (root / path).exists()}}


def main(argv=None):
    from phase3_engine import GameConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent worker processes, each sharing one warm planner")
    parser.add_argument("--replay-checks", choices=("all", "none"), default="all")
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/phase3"))
    parser.add_argument("--logs", type=Path, help="Optional privileged per-match JSONL directory")
    parser.add_argument("--seed", type=int, help="Run a single exact seed (overrides count/start)")
    args = parser.parse_args(argv)
    if args.matches < 1 or args.workers < 1:
        parser.error("--matches and --workers must be positive")
    config = asdict(GameConfig())
    seeds = [args.seed] if args.seed is not None else range(args.start_seed, args.start_seed + args.matches)
    jobs = [(seed, config, args.replay_checks == "all", args.logs) for seed in seeds]
    provenance = _provenance()
    args.output.mkdir(parents=True, exist_ok=True)
    journal = (args.output / 'progress.jsonl.tmp').open('w', encoding='utf-8')
    rows, started = [], perf_counter()
    executor = None
    if args.workers == 1:
        _initialize_worker()
        results = map(_run_job, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize_worker,
                                       mp_context=multiprocessing.get_context("spawn"))
        results = executor.map(_run_job, jobs, chunksize=1)
    try:
        for index, row in enumerate(results, 1):
            rows.append(row)
            journal.write(json.dumps(row, sort_keys=True) + '\n')
            journal.flush()
            if index % 25 == 0 or index == len(jobs) or row["crashes"]:
                print(f"{index}/{len(jobs)} seed={row['seed']} {row['reason'] or row['error']} "
                      f"elapsed={perf_counter()-started:.1f}s", flush=True)
    finally:
        journal.close()
        if executor is not None:
            executor.shutdown()
    summary = summarize(rows, config, perf_counter() - started)
    summary.update(provenance)
    summary['source_unchanged_during_run'] = provenance['source_sha256'] == _provenance()['source_sha256']
    if not summary['source_unchanged_during_run']:
        summary['acceptance'] = 'FAIL_SOURCE_CHANGED'
    summary["workers"] = args.workers
    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if key != "traceback"})
    with (args.output / "matches.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failures = [row for row in rows if row["crashes"] or row.get("deterministic_replay_failures")]
    if failures:
        (args.output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"acceptance": summary["acceptance"], "totals": summary["totals"],
                      "winners": summary["winners"], "win_reasons": summary["win_reasons"],
                      "means": summary["means"]}, indent=2), flush=True)
    return 0 if summary["acceptance"] in ("PASS", "SUBSET_ONLY") else 1


if __name__ == "__main__":
    from phase3_cli import use_project_environment
    use_project_environment()
    raise SystemExit(main())
