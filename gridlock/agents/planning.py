"""Shared plumbing for the planners (A* and CSP): plan at reset, follow the plan, and replan when
the world stops matching what the plan predicted.

Planners look ahead through the GridWorld hooks (get_state, simulate, legal_actions). They model
every other agent as standing still, as gridlock.eval.levelcheck does. When an opponent moves into
the way, the planner's own position or the objects end up somewhere the plan didn't predict, and
the planner replans from the state it is actually in.
"""

from dataclasses import replace
from typing import Optional

from gridlock.agents.base import Agent
from gridlock.env.grid import Action


class PlanningAgent(Agent):
    """Subclasses implement make_plan(env), which plans from env.get_state()."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.plan: list[Action] = []
        self.env = None
        self.replans = 0                       # times the plan was recomputed after reset
        self._predicted: list[tuple] = []      # own view after each planned action
        self._expect: Optional[tuple] = None   # own view expected before the next act()

    def make_plan(self, env) -> list[Action]:
        raise NotImplementedError

    def reset(self, env) -> None:
        self.env = env
        self.replans = 0
        self._replan()

    def act(self, obs) -> Action:
        if self._expect is not None and self.env is not None and self._view(self.env.get_state()) != self._expect:
            self.replans += 1
            self._replan()
        if not self.plan:
            self._expect = None
            return Action.STAY
        self._expect = self._predicted.pop(0)
        return self.plan.pop(0)

    # --- helpers -------------------------------------------------------------------------------

    def _replan(self) -> None:
        state = self.env.get_state()
        self.plan = list(self.make_plan(self.env))
        self._predicted = []
        for action in self.plan:
            state, _, _ = self.env.simulate(state, {self.agent_id: action})
            self._predicted.append(self._view(state))
        self._expect = self._view(self.env.get_state())

    def _view(self, state) -> tuple:
        """What the plan controls and predicts: own position, the objects, and whether finished."""
        return dict(state.agents).get(self.agent_id), state.objects, self.agent_id in state.done


def untimed(state):
    """The state with its clock zeroed. Neither the rules nor R read the clock, so transitions can
    be cached on this key while the search tracks time separately."""
    return replace(state, t=0)
