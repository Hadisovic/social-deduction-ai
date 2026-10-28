"""Cooperative external-controller boundary: only immutable actor-local values."""
from typing import Protocol
from among_us_map import get_map
from social_deduction.actor import Role,Phase
from social_deduction.phase3_api import Phase3Observation,Intent,IntentKind,intent_allowed,validate_observation


class ImpostorController(Protocol):
    def decide(self,observation:Phase3Observation)->Intent:...


class LegalImpostor:
    """No game/role roster/crew memory/replay is given to the implementation.

    This is an interface guarantee for cooperative code, not a Python sandbox.
    A fresh implementation is constructed for each match.
    """
    def __init__(self,implementation,player_id):
        if not callable(getattr(implementation,'decide',None)):raise TypeError('Controller requires decide(observation)')
        self.implementation=implementation;self.player_id=player_id;self.calls=0

    def decide(self,observation):
        validate_observation(observation)
        if observation.own.role is not Role.IMPOSTOR or observation.own.player_id!=self.player_id:
            raise ValueError('Impostor controller cannot change roles or owners')
        if not observation.own.active or observation.context.phase is Phase.FINISHED:
            raise ValueError('No living controller calls after death/termination')
        action=self.implementation.decide(observation);self.calls+=1
        if type(action) is not Intent:raise TypeError('Return the shared typed Intent, not a legacy action index')
        if action.kind is IntentKind.WAIT:return action  # harmless omission in any phase
        if not intent_allowed(observation,action):raise ValueError('External controller proposed an illegal/stale action')
        if action.kind is IntentKind.GO_TO_LOCATION and not get_map().contains((action.position.x,action.position.y)):
            raise ValueError('External controller proposed a location outside public walkable geometry')
        return action
