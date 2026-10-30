"""Versioned Classic coordinator; historical actor architecture and PPO unchanged."""
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
import torch
from jane.impostor.v1.features import OwnFeedback
from jane.impostor.v1.session import Proxy,resolve_team_contention
from jane.impostor.v1.rewards import ready_seconds
from .audit import Actor,JointRecorder,AuditTrail


def run_match(policy,crew_model,profile,seed,**kwargs):
    audit=AuditTrail(kwargs.get('output'))
    audit.write('header',schema='lizbon-decision-audit-v2',seed=seed,
        host_profile=profile.to_dict(),host_profile_sha256=profile.fingerprint())
    try:
        return _run_match(policy,crew_model,profile,seed,audit=audit,**kwargs)
    except BaseException as error:
        audit.write('failure',error=type(error).__name__,message=str(error))
        raise
    finally:audit.close()


def _run_match(policy, crew_model, profile, seed, *, output=None, audit=None,
              max_wall_seconds=300, max_observations=None, deterministic=False, status=None, live_dir=None,
              policy_update=0,mode='engineering'):
    from jane.classic_v2.engine import make_match,experiment
    settings=profile.settings;config=experiment(profile)
    from patrick_jane.network import collate
    from patrick_jane.actions import choose
    from navigation_service import NavigationPlanner
    from phase3_engine import stream_seed
    began = perf_counter()
    native = make_match(seed, config)
    native.opponent_regime = 'lizbon-classic-learned-v2'
    impostors = tuple(pid for pid in native.game.players if pid not in native.crew)
    public_planner = NavigationPlanner(native.game.base_map,cache_size=256)
    proxies = {pid:Proxy(Actor(pid,policy,settings,public_planner,stream_seed(seed,'jane-imp:'+pid),deterministic))
               for pid in impostors}
    native.opponents = proxies  # No tactical/script decide is called in this new regime.
    native.game._record('controller_override',channel='spectator-private',regime='lizbon-classic-learned-v2',
                        controllers={pid:'frozen-patrick-jane-102' if pid in native.crew else 'Lizbon-Classic-v2' for pid in native.game.players})
    metadata = dict(session='Lizbon-Classic-v2', source='upstream-d459f386+classic-v2',
        architecture=json.dumps(policy.architecture()),policy_update=policy_update,run_mode=mode,
        host_profile=profile.to_dict(),host_profile_sha256=profile.fingerprint(),
        policy_state='fresh untrained initialization' if policy_update==0 else 'trained checkpoint',
        controllers={pid:'frozen-patrick-jane-102' if pid in native.crew else 'Lizbon-Classic-v2' for pid in native.game.players})
    if live_dir:
        from patrick_jane.live import LiveBridge
        class JointLive(LiveBridge):
            def check(self):
                super().check()
                if self.recorder:
                    self.recorder.recorder.header.update(controller_types=metadata['controllers'],
                        game_schema='lizbon-classic-v2',training_session='Lizbon-Classic-v2',model_architecture=metadata['architecture'])
        native.live = JointLive(native,live_dir,dict(metadata,environment=0))
    observer = JointRecorder(native, output, metadata, native.live,audit=audit) if output else None
    if observer:
        native.recorder = observer
        for proxy in proxies.values(): proxy.observer = observer.impostor
    reward_components = {pid:Counter() for pid in impostors}; reward_done = False;contentions=Counter()
    original_step = native.game.step
    def observed_step(actions=None):
        nonlocal reward_done
        game = native.game; before_time = game.time; before_phase = game.phase.value
        before = {pid:(game.players[pid].active,game.players[pid].cooldown,pid in game.vented,
                       game.time<game.vent_transition.get(pid,0.)) for pid in impostors}
        cursor = len(game.event_log)
        resolved,declined=resolve_team_contention(actions)
        for pid,request in declined.items():
            contentions[request['kind']]+=1
            game._record('team_resource_contention',channel='spectator-private',player_id=pid,
                         request_kind=request['kind'],request_target=request['target'])
        result = original_step(resolved)
        elapsed = game.time-before_time
        events = game.event_log[cursor:]
        audit.write('effects',input_scope='offline-world-only',tick=game.tick,time=game.time,
            elapsed=elapsed,phase=game.phase.value,events=events,
            impostors={pid:dict(active=game.players[pid].active,position=game.players[pid].position,
                cooldown=game.players[pid].cooldown,vented=pid in game.vented,
                fake=game.players[pid].interaction_task,velocity=game.players[pid].velocity)
                for pid in impostors})
        killers = {e['killer_id'] for e in events if e['kind']=='elimination'}
        kills = sum(e['kind']=='elimination' for e in events)
        components = {pid:{} for pid in impostors}
        for pid in impostors:
            if kills: components[pid]['team_kill'] = float(kills)
            active, cooldown, vented, transition = before[pid]
            ready=ready_seconds(elapsed,cooldown,active_before=active,active_after=game.players[pid].active,
                roaming_before=before_phase=='roaming',roaming_after=game.phase.value=='roaming',
                vented_before=vented,vented_after=pid in game.vented,killed=pid in killers,
                transition_before=transition,transition_after=game.time<game.vent_transition.get(pid,0.),
                kill_cooldown=game.config.kill_cooldown)
            if ready:components[pid]['cooldown_ready_time']=-.005*ready
        if game.is_terminal and game.result.winner is not None and not reward_done:
            amount = 10. if game.result.winner.value=='impostor' else -10.
            for pid in impostors: components[pid]['team_outcome'] = amount
            reward_done = True
        for pid, values in components.items():
            if values and proxies[pid].rollout:
                proxies[pid].rollout[-1].reward += sum(values.values())
                reward_components[pid].update(values)
                if observer: observer.reward(game.time,pid,values)
                if native.live: native.live.reward(game.time,pid,values,sum(reward_components[pid].values()))
        return result
    native.game.step = observed_step
    response = native.prepare()
    crew_hidden = {pid:torch.zeros(1,256) for pid in native.crew}
    crew_rng = {pid:torch.Generator().manual_seed(stream_seed(seed,'jane-frozen-crew:'+pid) % (2**63-1)) for pid in native.crew}
    crew_choices = 0;crew_seconds=0.; policy.eval()
    while not response['terminal']:
        count = sum(len(p.rollout) for p in proxies.values())
        if perf_counter()-began >= max_wall_seconds or max_observations is not None and count >= max_observations:
            native.game.truncation_pending = dict(cause='engineering-or-wall-cut',simulation_seconds=native.game.time)
            native.game.finalize_truncation()
            if observer: observer.step(); observer.finish()
            if native.live: native.live.step(); native.live.publish(force=True)
            break
        for pid, proxy in proxies.items():
            p = native.game.players[pid]
            if not p.active: continue
            obs = native.observations[pid]
            task = p.interaction_task
            proxy.feedback = OwnFeedback(pid,obs.actor.match_id,obs.actor.tick,float(obs.actor.kill_cooldown),
                float(p.interaction_remaining) if task and task.startswith('fake:') else 0.,
                task[5:] if task and task.startswith('fake:') else None)
        ids = list(native.packets); decisions = {}
        if ids:
            inference_started=perf_counter()
            with torch.inference_mode():
                outputs, states = crew_model(collate([native.packets[p] for p in ids]),torch.cat([crew_hidden[p] for p in ids]))
                for i,pid in enumerate(ids):
                    crew_hidden[pid] = states[i:i+1]
                    out = {k:v[i] for k,v in outputs.items()}
                    choice,lp,entropy,probs = choose(out,native.packets[pid],generator=crew_rng[pid])
                    decisions[pid] = dict(choice=choice,logp=float(lp),value=float(out['value']),
                                          entropy=float(entropy),probabilities=probs,update=102)
            crew_choices += len(ids)
            crew_seconds+=perf_counter()-inference_started
        response = native.advance(decisions)
        if status and count % 100 < 2:
            status(dict(seed=seed,simulation_seconds=native.game.time,impostor_observations=count,
                        crew_choices=crew_choices,wall_seconds=perf_counter()-began))
    summary = native.summary()
    summary.update(host_profile=profile.to_dict(),host_profile_sha256=profile.fingerprint(),
                   impostor_actions={pid:dict(proxy.counts) for pid,proxy in proxies.items()},
                   impostor_rewards={pid:dict(c) for pid,c in reward_components.items()},
                   impostor_observations=sum(len(p.rollout) for p in proxies.values()),
                   team_resource_contentions=dict(contentions),
                   timing=dict(crew_model_and_sampling_seconds=crew_seconds,
                               impostor_features_seconds=sum(p.timing['feature_seconds'] for p in proxies.values()),
                               impostor_model_and_sampling_seconds=sum(p.timing['inference_seconds'] for p in proxies.values())),
                   impostor_reward_regime='jane-impostor-team-v1',
                   wall_seconds=perf_counter()-began,controllers=metadata['controllers'])
    if output:
        Path(output).with_suffix('.result.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary, [proxy.rollout for proxy in proxies.values()]
