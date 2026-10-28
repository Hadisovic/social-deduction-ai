"""Legal engineering choreography, explicitly separate from natural gameplay."""
from time import perf_counter

from social_deduction.actor import Phase
from social_deduction.phase3_api import Intent, IntentKind, ClaimDraft, ClaimKind, intent_allowed
from crewmate_v2.fourcrew.scenarios import ScenarioMatch
from crewmate_v2.config import LobbySettings, MatchMixture


CASES=('self_report','leave_unreported','innocent_report','frame_innocent','unwitnessed',
       'direct_witness','weak_evidence','strong_evidence','contradictory_claims','multiple_suspects',
       'new_evidence_after_defense','profile_variation')


def evidence_fixture(seed,case,*,recorder_factory=None):
    if case not in CASES:raise ValueError('Unknown engineering case')
    match=ScenarioMatch(seed,context='ordinary',lobby=LobbySettings(),mixture=MatchMixture(1.),deadline=perf_counter()+90)
    recorder=recorder_factory(match) if recorder_factory else None
    match._prefix_record=recorder
    if recorder:recorder.after_step()
    game=match.game;site=tuple(map(float,next(d.standing for d in game.map.destinations if d.name=='EmergencyConsole')))
    victim=match.crew[0];others=list(match.crew[1:]);observer=others[0]
    rooms=('Weapons','Admin','MedBay')
    positions=[tuple(map(float,next(d.standing for d in game.map.destinations if d.room==room and d.category=='task'))) for room in rooms]
    targets={pid:xy for pid,xy in zip(others,positions)}
    witness=others[-1] if case=='new_evidence_after_defense' else observer
    if case in ('direct_witness','strong_evidence','new_evidence_after_defense'):targets[witness]=site
    targets.update({victim:site,match.impostor:site});match._drive(targets,225)
    while game.tick<225:match._prefix_tick()
    kill=next(a for a in game.observe(match.impostor).actions if a.kind is IntentKind.KILL and a.target==victim)
    match._prefix_tick({match.impostor:kill})
    if case in ('leave_unreported','unwitnessed','profile_variation'):
        match._drive({match.impostor:positions[2]},375)
    else:
        match._drive({pid:site for pid in (*others,match.impostor)},375)
    while game.tick<375:match._prefix_tick()
    if case not in ('leave_unreported','unwitnessed','profile_variation'):
        reporter=match.impostor if case=='self_report' else observer
        report=next(a for a in game.observe(reporter).actions if a.kind is IntentKind.REPORT and a.target==victim)
        match._prefix_tick({reporter:report})
    match.prefix=dict(version='phase2-legal-engineering-v1',context=case,handoff_tick=game.tick,prefix_reports=int(game.phase is not Phase.ROAMING),prefix_deaths=1)
    return match,recorder,dict(victim=victim,observer=observer,witness=witness,impostor=match.impostor,innocent=others[1])


def discussion_messages(match,messages,recorder=None):
    """Each message waits for its actual speaker slot; no forced learned actions.

    This function is scripted engineering setup only. Caller must never install it
    in a learned controller. Claims use public IDs/regions and permitted timing.
    """
    game=match.game;match._prefix_record=recorder
    remaining=dict(messages);accepted=[]
    while game.phase is Phase.DISCUSSION:
        speaker=game.observe(match.crew[-1]).context.speaker_id
        actions={}
        if speaker in remaining:
            kind,target=remaining.pop(speaker);obs=game.observe(speaker)
            claim=Intent(IntentKind.CLAIM,claim=ClaimDraft(kind,target,obs.own.room,obs.tick))
            if not intent_allowed(obs,claim):raise ValueError('Nonlegal diagnostic claim')
            actions[speaker]=claim
        mark=len(game.event_log);match._prefix_tick(actions)
        accepted.extend(e for e in game.event_log[mark:] if e['kind']=='claim')
    if remaining:raise ValueError('Engineering speaker coverage failed')
    return accepted
