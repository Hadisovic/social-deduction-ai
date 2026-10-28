"""Freeze releases and audit domain-separated identity/color/spawn associations."""
import argparse
from collections import Counter
import json
from random import Random
from phase3_engine import stream_seed
from social_deduction.truth import assign_identities, assign_roles
from social_deduction.actor import Role
from .config import BASELINE_SHA, FROZEN, PARTITIONS, source_hashes, verify_frozen


def audit():
    verify_frozen()
    associations = {key: Counter() for key in ('public_id', 'color', 'spawn_slot')}
    for seed in range(PARTITIONS['train'][0], PARTITIONS['train'][0] + 1000):
        identities = assign_identities(5, seed=stream_seed(seed, 'identities'))
        roles = assign_roles(5, seed=stream_seed(seed, 'roles'))
        colors = ['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Orange', 'Black', 'Pink', 'White', 'Brown']
        Random(stream_seed(seed, 'colors')).shuffle(colors)
        for slot, (identity, role, color) in enumerate(zip(identities, roles, colors)):
            if role is Role.IMPOSTOR:
                associations['public_id'][identity.player_id] += 1
                associations['color'][color] += 1
                associations['spawn_slot'][str(slot)] += 1
    if any(len(associations[k]) != n or min(associations[k].values()) == 0 for k, n in
           (('public_id', 5), ('color', 10), ('spawn_slot', 5))):
        raise ValueError('Impostor association coverage failed')
    return dict(baseline_sha=BASELINE_SHA, frozen_sha256=FROZEN, partitions=PARTITIONS,
                sources=source_hashes(), impostor_counts=dict(associations), matches_audited=1000,
                notice='Counts are a correlation diagnostic, not a proof of independence. Role/identity/color RNG domains are separate. Opponent types and all seeds stay trainer-only.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(audit(), indent=2))
