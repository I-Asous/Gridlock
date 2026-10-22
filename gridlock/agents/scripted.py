"""Scripted opponents and reference policies."""

import random

from gridlock.agents.base import Agent
from gridlock.env.grid import Action


class RandomAgent(Agent):
    def __init__(self, agent_id: str, seed: int | None = None):
        super().__init__(agent_id)
        self.rng = random.Random(seed)

    def act(self, obs) -> Action:
        return self.rng.choice(list(Action))


class GreedyAgent(Agent):
    """Moves toward the goal along the shortest walkable path, ignoring others."""

    def act(self, obs) -> Action:
        raise NotImplementedError
