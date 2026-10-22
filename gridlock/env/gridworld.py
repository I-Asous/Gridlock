"""Gym-style multi-agent environment. Exposes R only; R* lives in gridlock.intent."""

from dataclasses import dataclass, field
from typing import Any, Optional

from gridlock.env.events import Event, Pos
from gridlock.env.grid import Action, Grid
from gridlock.env.referee import Referee


@dataclass(frozen=True)
class State:
    t: int
    agents: tuple[tuple[str, Pos], ...]
    objects: tuple[tuple[str, Any], ...]      # object id -> position / status
    flags: frozenset[str] = field(default_factory=frozenset)  # e.g. "sensor_off"


class GridWorld:
    def __init__(self, scenario, max_steps: int = 100):
        self.scenario = scenario
        self.grid: Grid = scenario.grid
        self.referee = Referee(self.grid)
        self.reward_fn = scenario.reward_fn
        self.max_steps = max_steps
        self.state: Optional[State] = None

    def reset(self, seed: Optional[int] = None) -> tuple[dict[str, Any], dict]:
        raise NotImplementedError

    def step(self, actions: dict[str, Action]):
        """Returns obs, rewards, terminated, truncated, info (info["events"]: list[Event])."""
        raise NotImplementedError

    def observe(self, agent_id: str) -> Any:
        raise NotImplementedError

    # Lookahead hooks for the A* and CSP planners.
    def get_state(self) -> State:
        return self.state

    def set_state(self, state: State) -> None:
        self.state = state

    def legal_actions(self, state: State, agent_id: str) -> list[Action]:
        return self.referee.legal_actions(state, agent_id)

    def simulate(self, state: State, actions: dict[str, Action]) -> tuple[State, dict[str, float], list[Event]]:
        """Pure transition: no side effects on self.state."""
        raise NotImplementedError
