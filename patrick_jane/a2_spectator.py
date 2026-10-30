"""Observer-only mechanics snapshots; never passed to policy inference."""
def state(game):
    return dict(doors={key:end for key,end in game.door_until.items() if key in game.map.closed_doors},
        vented=dict(game.vented),vent_positions={key:list(v['position']) for key,v in game.vents.items()},
        local_doors={pid:sorted(keys) for pid,keys in game.door_known.items()},
        visible={pid:[p.player_id for p in game.observe(pid).visible_players] for pid,p in game.players.items() if p.active and pid not in game.vented},
        mechanics_regime=game.experiment.mechanics_regime,opponent_regime=game.experiment.opponent_regime,
        reward_regime=game.experiment.reward_regime)
