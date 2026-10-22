"""A* planner: least-cost path under R (cost = -R per transition)."""

from gridlock.agents.base import Agent
from gridlock.env.grid import Action


class AStarAgent(Agent):
    def __init__(self, agent_id: str, max_expansions: int = 100_000):
        super().__init__(agent_id)
        self.max_expansions = max_expansions
        self.plan: list[Action] = []

    def reset(self, env) -> None:
        self.plan = self.search(env)

    def search(self, env) -> list[Action]:
        raise NotImplementedError

    def heuristic(self, state, env) -> float:
        raise NotImplementedError

    def act(self, obs) -> Action:
        return self.plan.pop(0) if self.plan else Action.STAY
