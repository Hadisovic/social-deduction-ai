"""Metadata-backed, role-independent, unique-per-owner task allocation."""
from dataclasses import replace
import hashlib
from random import Random

ALIASES={'EmptyChute':'EmptyGarbage'}


def family(kind):return ALIASES.get(kind,kind)


def pools(world):
    from crewmate_v2.tasks import task_catalog
    catalog=task_catalog(world);result={category:{} for category in ('Common','Long','Short')}
    for task in world.data['tasks'].values():
        kind=task['taskType'];category=task['length']
        if kind=='VentCleaning' or kind not in catalog:continue
        result[category].setdefault(family(kind),set()).add(kind)
    return {category:{key:tuple(sorted(kinds)) for key,kinds in sorted(values.items())}
            for category,values in result.items()}


def assignments(world,seed,roster,counts):
    from phase3_engine import stream_seed
    from crewmate_v2.tasks import task_catalog,make_plan,OwnedTask
    category_pools=pools(world);catalog=task_catalog(world)
    for category,count in zip(('Common','Long','Short'),counts):
        if count>len(category_pools[category]):raise ValueError(f'Not enough unique {category} types')
    common_rng=Random(stream_seed(seed,'classic-v2:common'))
    common_types=common_rng.sample(list(category_pools['Common']),counts[0])
    common_plans=[make_plan('',common_rng.choice(category_pools['Common'][kind]),'Common',catalog,common_rng)
                  for kind in common_types]
    result={}
    for pid in sorted(roster):
        rng=Random(stream_seed(seed,'classic-v2:tasks:'+pid))
        specs=[replace(plan,owner_id=pid) for plan in common_plans]
        used={family(p.task_type) for p in specs}
        for category,count in zip(('Long','Short'),counts[1:]):
            available=[key for key in category_pools[category] if key not in used]
            if len(available)<count:raise ValueError(f'Not enough non-overlapping {category} types')
            for key in rng.sample(available,count):
                kind=rng.choice(category_pools[category][key])
                specs.append(make_plan(pid,kind,category,catalog,rng));used.add(key)
        result[pid]=tuple(OwnedTask('classic2-'+hashlib.sha256(
            f'{stream_seed(seed,"classic-v2:ids")}:{pid}:{i}'.encode()).hexdigest()[:20],plan)
            for i,plan in enumerate(specs))
        if len({family(t.plan.task_type) for t in result[pid]})!=sum(counts):
            raise AssertionError('Duplicate assigned task family')
    return result
