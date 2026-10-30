"""Pure actor action decoder, usable in simulation workers without Torch."""
from social_deduction.phase3_api import Intent,IntentKind
from .api import Action,Event
from .features import goal_for


def execute(choice,packet,nav):
    obs=packet.observation;a=obs.actor;now=a.time_seconds
    target=packet.candidates[choice.target] if choice.target>=0 else None
    if packet.phase==0:
        interaction=choice.interaction
        if interaction==0:
            return nav.command(choice.navigation,goal_for(target) if target else None,now,obs)
        if interaction==1:
            if target.key in a.interactable_tasks:return Intent(IntentKind.INTERACT,target.key)
            if getattr(packet,'task_pursuit_mode',False):
                # Same sampled task/target heads; no scripted interaction on arrival.
                if target.detail.console_id in a.interactable_destinations:return Intent(IntentKind.WAIT)
                if nav.goal is not None and nav.goal.key==target.key and a.navigation_status=='moving':return nav.command(0,None,now,obs)
                return nav.command(2,goal_for(target),now,obs)
            return Intent(IntentKind.INTERACT,target.key)
        if interaction==2:return Intent(IntentKind.REPORT,target.key)
        if interaction==3:return Action('repair',target.key)
        if interaction==4:return Intent(IntentKind.CALL_MEETING,target.key)
    elif packet.phase==1:
        if choice.communication==0:return Intent(IntentKind.WAIT)
        kinds=('','share_evidence','accuse','defend','question','strategic_claim')
        event=target.detail if target.kind=='evidence' else None
        subject=event.subject if event else target.key if target.kind=='player' else a.own.player_id
        room=event.room if event else getattr(target.detail,'room',a.own.room)
        # Claimed time and location remain assertions, including strategic false claims.
        statement=Event(kinds[choice.communication],subject,event.other if event else '',room,
            target.position,event.observed if event else now,now,a.own.player_id,'claim',
            event.kind if event else '',(event.origin_source or event.source) if event else '')
        return Action('communication',statement=statement)
    elif packet.phase==2:
        return Intent(IntentKind.SKIP) if choice.voting==14 else Intent(IntentKind.VOTE,packet.other_ids[choice.voting])
    raise ValueError('Unsupported phase/action')
