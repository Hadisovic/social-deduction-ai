"""Trusted attachment of independent crew observers to an unchanged scripted match."""
from belief.memory import ActorMemory
from social_deduction.actor import Role


class MatchObserver:
    def __init__(self, match, model):
        self.model = model
        self.memories = {}
        self.beliefs = {}
        self.active = {}
        for pid in sorted(match.controllers):
            obs = match.game.observe(pid)
            if obs.own.role is Role.CREWMATE:
                self.memories[pid] = ActorMemory()
        self.focal = next(iter(self.memories))
        self.update(match)

    def update(self, match):
        if match.game.is_terminal:
            # The final action may kill this observer. Update its own UI status
            # without admitting any terminal packet to memory or inference.
            for pid in self.memories:
                self.active[pid] = match.game.observe(pid).own.active
            return
        if match.game.tick % match.decision_ticks:
            return
        for pid,memory in self.memories.items():
            observation = match.game.observe(pid)
            self.active[pid] = observation.own.active
            if observation.own.active and observation.tick > memory.tick:
                memory.update(observation)
                self.beliefs[pid] = self.model.predict(memory)

    def cycle(self):
        ids = tuple(self.memories)
        self.focal = ids[(ids.index(self.focal)+1) % len(ids)]

    def step(self, match):
        self.update(match)
        result = match.step()
        self.update(match)
        return result
