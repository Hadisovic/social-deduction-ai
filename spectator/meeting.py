"""Public current-meeting projection; historical submissions never reveal ballots."""


def current_meeting(frame):
    start=next((e for e in reversed(frame.events_up_to_now) if e.get('kind')=='meeting_started'),None)
    if start is None:return dict(participants=[],claims=[],votes=[])
    mid=start.get('meeting_id');tick=start.get('tick',0)
    eligible={pid for pid in start.get('participants',[]) if pid in frame.players and frame.players[pid].status=='alive'}
    events=[e for e in frame.events_up_to_now if e.get('tick',0)>=tick and
            (e.get('meeting_id')==mid or e.get('meeting_id') is None)]
    claims=[e for e in events if e.get('kind') in ('claim','structured_claim') and e.get('speaker_id') in eligible]
    published={e['voter_id']:e for e in events if e.get('kind')=='vote' and e.get('channel')=='public' and e.get('voter_id') in eligible}
    submitted={e['voter_id'] for e in events if e.get('kind')=='vote_submitted' and e.get('meeting_id')==mid and e.get('voter_id') in eligible}
    votes=[dict(voter_id=pid,published=pid in published,target_id=published[pid].get('target_id') if pid in published else None) for pid in sorted(submitted|published.keys())]
    return dict(participants=sorted(eligible),claims=claims,votes=votes)
