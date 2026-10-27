"""Read completed engineering replays and render an honest selected-actor debug frame."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--spectator-source',type=Path,required=True);a=p.parse_args()
    sys.path.append(str(a.spectator_source.resolve()))
    from spectator.replay_format import parse_replay
    from spectator.playback import ReplayPlayer
    from spectator.meeting import current_meeting
    rows=[];selected=None
    for path in sorted((ROOT/'local-runs/representative-benchmark/replays').rglob('*.jsonl')):
        data=parse_replay(path);assert data['complete']
        header=data['header'];context=header['config']['context']
        assert len(header['controller_types'])==5 and list(header['controller_types'].values()).count('ppo')==4
        expected=json.loads((ROOT/'local-runs/representative-benchmark'/('match-'+str(('ordinary','body','meeting_crew','meeting_impostor').index(context))+'.json')).read_text())
        assert data['result']['reason']==expected['reason'] and len(data['neural'])==expected['decisions']
        actors={n['player_id'] for n in data['neural']}
        assert len(actors)==(4 if context=='ordinary' else 3)
        for neural in data['neural']:
            assert abs(sum(neural['action_probabilities'].values())-1)<1e-5
            assert neural['sim_time']>=header['config']['prefix_handoff_seconds']-1e-8
            assert neural['suspicion_probabilities'] and neural['selected_action_index']>=0
        dead=set();phase='roaming';meeting_completions=dead_actions=0
        for line in path.open(encoding='utf-8'):
            r=json.loads(line)
            if r.get('record')=='neural':dead_actions+=r['player_id'] in dead
            if r.get('record')!='event':continue
            kind=r.get('kind')
            if kind=='elimination':dead.add(r['victim_id'])
            if kind=='ejection' and r.get('player_id'):dead.add(r['player_id'])
            if kind=='meeting_started':phase='discussion'
            if kind=='voting_started':phase='voting'
            if kind=='exploration_resumed':phase='roaming'
            if kind=='vote_submitted':assert r['voter_id'] not in dead
            if kind=='claim':assert r['speaker_id'] not in dead
            meeting_completions+=kind=='task_completed' and phase in ('discussion','voting')
        assert dead_actions==meeting_completions==0
        frozen_pairs=0
        for before,after in zip(data['ticks'],data['ticks'][1:]):
            if before['phase'] in ('discussion','voting') and after['phase'] in ('discussion','voting'):
                for pid,state in after['players'].items():
                    prior=before['players'][pid]
                    assert state['position']==prior['position']
                    assert state['ghost_position']==prior['ghost_position']
                    assert state['tasks_completed']==prior['tasks_completed']
                frozen_pairs+=1
        player=ReplayPlayer(data)
        for frame in data['ticks']:
            if frame['phase']=='voting':
                player.seek_to_time(frame['sim_time']);meeting=current_meeting(player.current_state())
                assert all(v['published'] or v['target_id'] is None for v in meeting['votes'])
        rows.append(dict(context=context,neural_actor_identities=len(actors),recorded_neural_decisions=len(data['neural']),
            outcome_matches_simulator=True,no_dead_policy_or_ballot_actions=True,meeting_task_completions=0,
            frozen_meeting_frame_pairs=frozen_pairs,scripted_prefix_not_presented_as_neural=True))
        if context=='meeting_crew':selected=(data,path)
    os.environ['SDL_VIDEODRIVER']='dummy'
    import pygame
    from among_us_map import get_map
    from spectator.app_state import SpectatorAppState
    from spectator.ui_renderer import SpectatorRenderer
    pygame.init();data,path=selected;player=ReplayPlayer(data)
    time=next(t['sim_time'] for t in data['ticks'] if t['phase']=='voting');player.seek_to_time(time)
    actor=next(n['player_id'] for n in reversed(data['neural']) if n['sim_time']<=time)
    status=json.loads((ROOT/'local-runs/live-watch/fourcrew-reporting-2048-seed17-v1/status.json').read_text())
    app=SpectatorAppState(frame_state=player.current_state(),episode_info=player.episode_info(),selected_player=actor,
        current_replay_path=str(path),player_duration=player.duration,training_metrics=status)
    app.view_options['private_debug']=True
    renderer=SpectatorRenderer(get_map(),(1700,1100));image=renderer.draw(app)
    assert len(app._speed_hitboxes)==7
    pygame.image.save(image,str(ROOT/'local-runs/readiness-meeting.png'));pygame.quit()
    report=dict(status='verified_actual_engineering_replays',checks=rows,
                rendered='headless selected-actor private debug; no visible desktop launch claimed',
                viewer_independent_of_training=True,study_started=False)
    with (ROOT/'local-runs/spectator-readiness.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
