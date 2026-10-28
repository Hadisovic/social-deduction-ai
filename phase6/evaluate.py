"""Paired development evaluation. Reserved validation/final seeds stay closed."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from phase5_analysis import summarize
from phase5_policy import StrategicPolicy
from social_deduction.actor import Role
from .config import BASELINE_SHA, Experiment, ROOT, digest, load_experiment, seeds, source_hashes, verify_frozen
from .env import Phase6Env
from .train import load_policy


def evaluate(output, experiment, *, matches=2, checkpoints=(), include_release=True):
    """Same new rules and paired dev seeds for all controls, not historical wins.

    This first milestone deliberately exposes no validation/final-test CLI. A
    reviewed budget/selection runner will open those partitions after approval.
    """
    manifest = seeds('development', matches)
    verify_frozen()
    torch.set_num_threads(1)
    methods = [('idle', None), ('random', None), ('scripted', None)]
    if include_release:
        release, _ = StrategicPolicy.load(ROOT / 'artifacts/phase5/release/policy.pt')
        methods.append(('phase5_release_transfer', release))
    for path in checkpoints:
        policy, trained_experiment, _ = load_policy(path)
        if trained_experiment != experiment:
            raise ValueError('Checkpoint experiment differs from evaluation conditions')
        methods.append((str(path), policy))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = perf_counter()
    rows = []
    env = Phase6Env(experiment)
    for family in ('ID', 'patient'):
        env.heldout = family == 'patient'
        for method, policy in methods:
            for seed in manifest:
                packet, _ = env.reset(seed=seed)
                rng = np.random.default_rng(seed ^ 999)
                action_counts = Counter()
                ends = Counter()
                decisions = 0
                simulated = 0.
                while True:
                    if method == 'scripted':
                        intent = env.match.controllers[env.focal].decide(env.observation)
                        action_counts[intent.kind.value] += 1
                        # Original script timing: one 0.6s decision, no option hold.
                        packet, _, done, _, info = env.step_scripted(intent)
                    else:
                        action = 0 if method == 'idle' else int(rng.choice(np.flatnonzero(packet['mask']))) if policy is None else policy.act(packet, True)[0]
                        action_counts[env.options[action].kind] += 1
                        packet, _, done, _, info = env.step(action)
                    decisions += 1
                    simulated += info['elapsed_seconds']
                    ends[info['option_end']] += 1
                    if done:
                        break
                game = env.match.game
                votes = [e for e in game.event_log if e['kind'] == 'vote' and e.get('voter_id') == env.focal]
                cast = [e for e in votes if e.get('target_id') is not None]
                ejections = [e for e in game.event_log if e['kind'] == 'ejection' and e.get('player_id') is not None]
                correct_ejections = sum(game.players[e['player_id']].role is Role.IMPOSTOR for e in ejections)
                row = dict(method=method, family=family, seed=seed, **info['episode'],
                           votes_cast=len(cast), votes_correct=sum(game.players[e['target_id']].role is Role.IMPOSTOR for e in cast),
                           skips=len(votes) - len(cast), correct_ejections=correct_ejections,
                           innocent_ejections=len(ejections) - correct_ejections,
                           valid_reports=sum(e['kind'] == 'report' and e.get('reporter_id') == env.focal for e in game.event_log),
                           decisions=decisions, simulated_seconds=simulated,
                           action_counts=dict(action_counts), option_ends=dict(ends),
                           impostor_win=game.result.winner is Role.IMPOSTOR,
                           scripted_impostor_eliminations=game.metrics['eliminations'])
                rows.append(row)
                with (output / 'matches.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
    result = dict(status='development_diagnostic_only', baseline_sha=BASELINE_SHA,
                  experiment=experiment.to_dict(), seed_partition='development', match_seeds=manifest,
                  phase6_sources=source_hashes(), seconds=perf_counter() - began,
                  checkpoint_sha256={str(p): digest(p) for p in checkpoints},
                  release_sha256=digest(ROOT / 'artifacts/phase5/release/policy.pt') if include_release else None,
                  **summarize(rows),
                  notice='No strategic-gain claim from small dev checks. Release transfer is reevaluated under new rules; no learned impostor exists.')
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    verify_frozen()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matches', type=int, default=2, help='Per method per family; development partition only')
    parser.add_argument('--checkpoint', type=Path, action='append', default=[])
    args = parser.parse_args()
    result = evaluate(args.output, load_experiment(args.config), matches=args.matches, checkpoints=args.checkpoint)
    print(json.dumps({'status': result['status'], 'seconds': result['seconds']}))


if __name__ == '__main__':
    main()
