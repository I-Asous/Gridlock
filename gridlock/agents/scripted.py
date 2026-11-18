"""Scripted opponents and reference policies."""

import random

from gridlock.agents.base import Agent
from gridlock.env.events import Pos
from gridlock.env.grid import Action, Tile

MOVES = (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP)   # also the tie-break order


class RandomAgent(Agent):
    def __init__(self, agent_id: str, seed: int | None = None):
        super().__init__(agent_id)
        self.rng = random.Random(seed)

    def act(self, obs) -> Action:
        return self.rng.choice(list(Action))


class GreedyAgent(Agent):
    """Moves toward the goal along the shortest walkable path, ignoring others.

    reset() measures every cell's walking distance to the nearest goal tile; act() then steps
    to a neighbouring cell that is one step closer, and stays put once on a goal. Objects and
    other agents are ignored, so it walks straight through a vase in its path. The observation
    must start with the agent's own (x, y), as GridWorld.observe provides.
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.distance: dict[Pos, int] = {}

    def reset(self, env) -> None:
        self.distance = env.grid.distances_from(env.grid.cells(Tile.GOAL))

    def act(self, obs) -> Action:
        x, y = int(obs[0]), int(obs[1])
        here = self.distance.get((x, y))
        for action in MOVES:
            dx, dy = action.value
            there = self.distance.get((x + dx, y + dy))
            if here is not None and there is not None and there < here:
                return action
        return Action.STAY
