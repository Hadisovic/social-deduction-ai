"""Trusted legal setup and matched private-state clones. No actor gets truth."""
import copy
import math
from time import perf_counter
from crewmate_v2.fourcrew.scenarios import ScenarioMatch
from crewmate_v2.config import LobbySettings,MatchMixture
from crewmate_v2.phase2.scenarios import evidence_fixture
from social_deduction.phase3_observation import visible_between
from social_deduction.phase3_api import IntentKind


def clone_prefix(match):
    if match.policy_version is not None or match._prefix_record is not None:raise ValueError('Clone only an uncollected unrecorded prefix')
    shared=[match.game.planner,match.game.map,match.game._owners,*[l.roles for l in match.ledgers.values()],*[v.belief_model for v in match.views.values()]]
    memo={id(x):x for x in shared};memo[id(match.match._actor_hash)]=match.match._actor_hash.copy()
    result=copy.deepcopy(match,memo)
    if result.game is match.game or any(result.views[p].memory is match.views[p].memory for p in match.crew):raise ValueError('Private clone alias')
    if any(result.views[p].match.game is not result.game for p in result.crew):raise ValueError('Clone has stale game reference')
    return result


def social_fixture(seed,case):
    mapping={'empty':'weak_evidence','innocent':'innocent_report','impostor':'self_report',
             'direct':'direct_witness','conflicting':'weak_evidence','changed':'new_evidence_after_defense'}
    if case not in mapping:raise ValueError('Unknown social context')
    return evidence_fixture(seed,mapping[case])


WITNESS_CASES=('immediate','outside_range','wall_obstructed','spectator_only','multiple_race','dead_witness','unknown_killer')


def witness_fixture(seed,case,prekill_policy=None):
    if case not in WITNESS_CASES:raise ValueError('Unknown visibility fixture')
    m=ScenarioMatch(seed,context='ordinary',lobby=LobbySettings(),mixture=MatchMixture(1.),deadline=perf_counter()+90)
    g=m.game;victim=m.crew[0];witness=m.crew[1];others=list(m.crew[2:])
    points=[tuple(map(float,d.standing)) for d in g.map.destinations if d.category=='task']
    center=tuple(map(float,next(d.standing for d in g.map.destinations if d.name=='EmergencyConsole')))
    site=center;position=center
    if case=='outside_range':
        choices=[(center[0]+dx,center[1]+dy) for dx in (-3.,-2.,0.,2.,3.) for dy in (-3.,-2.,0.,2.,3.)]
        position=next(p for p in choices if g.map.contains(p) and g.lobby.engine_config(g.map.radius).kill_range<math.dist(p,site)
            and math.dist(p,site)>g.observe(witness).settings.report_range+.8 and visible_between(g.map,p,site,4.0))
    elif case=='wall_obstructed':
        # Navigation arrives in a legal area, not at an infinitely precise point.
        # Require occlusion to survive the full .45-unit arrival tolerance.
        def robust(a,b):
            if not 1.3<math.dist(a,b)<3.5 or visible_between(g.map,a,b,100):return False
            around=lambda p:[(p[0]+x,p[1]+y) for x in (-.5,0,.5) for y in (-.5,0,.5) if g.map.contains((p[0]+x,p[1]+y))]
            return all(not visible_between(g.map,x,y,100) for x in around(a) for y in around(b))
        site,position=next((a,b) for a in points for b in points if robust(a,b))
    elif case in ('spectator_only','unknown_killer'):position=max(points,key=lambda p:math.dist(p,site))
    far=sorted(points,key=lambda p:-math.dist(p,site))
    targets={m.impostor:site,victim:site,witness:position}
    targets.update({p:(site if case=='multiple_race' else far[i]) for i,p in enumerate(others)})
    m._drive(targets,225)
    if case=='wall_obstructed':
        a=g.players[witness].position;b=g.players[victim].position
        if math.dist(a,b)>4.5 or visible_between(g.map,a,b,4.5):raise ValueError('Actual arrived geometry is not wall-obstructed')
    while g.tick<224:m._prefix_tick()
    if prekill_policy is not None:
        import torch
        from crewmate_v2.learning.policy import tensors
        for pid,view in m.views.items():
            if not g.players[pid].active:continue
            view.refresh();packet=view._packet()
            with torch.no_grad():distribution,_=prekill_policy(tensors(packet))
            choice=view.options[int(distribution.probs[0].argmax())]
            m._lease(pid,choice)
    while g.tick<225:m._prefix_tick()
    kill=next(a for a in g.observe(m.impostor).actions if a.kind is IntentKind.KILL and a.target==victim)
    m._prefix_tick({m.impostor:kill})
    if case=='dead_witness':
        while g.tick<450:m._prefix_tick()
        second=next(a for a in g.observe(m.impostor).actions if a.kind is IntentKind.KILL and a.target==witness)
        m._prefix_tick({m.impostor:second})
    if case=='unknown_killer':
        m._drive({m.impostor:far[1]},375);m._drive({witness:site},450)
    m.prefix=dict(version='phase21-legal-visibility-v1',context=case,handoff_tick=g.tick,prefix_reports=0,
                  prefix_deaths=2 if case=='dead_witness' else 1)
    return m,dict(victim=victim,witness=witness,impostor=m.impostor)
