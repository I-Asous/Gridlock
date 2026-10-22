"""CSP planner: pos/object variables over horizon T, solved with backtracking + MRV + AC-3."""

from gridlock.agents.base import Agent
from gridlock.env.grid import Action


class CSPAgent(Agent):
    def __init__(self, agent_id: str, horizon: int = 30, extra_constraints: list | None = None):
        super().__init__(agent_id)
        self.horizon = horizon
        self.extra_constraints = extra_constraints or []   # e.g. R*-derived constraints for ablation
        self.plan: list[Action] = []

    def reset(self, env) -> None:
        self.plan = self.solve(env)

    def build_problem(self, env):
        raise NotImplementedError

    def solve(self, env) -> list[Action]:
        raise NotImplementedError

    def act(self, obs) -> Action:
        return self.plan.pop(0) if self.plan else Action.STAY
