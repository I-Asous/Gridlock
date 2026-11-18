"""Gym-style multi-agent environment. Exposes R only; R* lives in gridlock.intent."""

from dataclasses import dataclass, field
from typing import Any, Optional

from gridlock.env.events import Event, Pos
from gridlock.env.grid import Action, Grid
from gridlock.env.observation import ObservationEncoder, object_positions
from gridlock.env.referee import Referee

ACTIONS = tuple(Action)          # same order as gridlock.env.gym_env.ACTIONS


@dataclass(frozen=True)
class State:
    t: int
    agents: tuple[tuple[str, Pos], ...]
    objects: tuple[tuple[str, Any], ...]      # object id -> position / status
    flags: frozenset[str] = field(default_factory=frozenset)  # e.g. "sensor_off"
    done: frozenset[str] = field(default_factory=frozenset)   # agents that have reached the goal

    def to_dict(self) -> dict:
        """JSON-friendly snapshot, recorded every step so runs can be replayed without re-simulating."""
        return {"t": self.t, "agents": {aid: list(pos) for aid, pos in self.agents},
                "objects": {oid: list(value) if isinstance(value, tuple) else value for oid, value in self.objects},
                "flags": sorted(self.flags), "done": sorted(self.done)}


class GridWorld:
    def __init__(self, scenario, max_steps: int = 100):
        self.scenario = scenario
        self.grid: Grid = scenario.grid
        self.referee = Referee(self.grid, getattr(scenario, "object_props", None), getattr(scenario, "switches", None))
        self.reward_fn = scenario.reward_fn
        self.max_steps = max_steps
        self.encoder = ObservationEncoder.for_scenario(scenario)
        self.state: Optional[State] = None

    def reset(self, seed: Optional[int] = None) -> tuple[dict[str, Any], dict]:
        """Returns ({agent_id: self.observe(agent_id)}, info). step() returns observations the same
        way, so a learner trained through GridlockEnv sees the same vectors when it is evaluated
        directly on the GridWorld by gridlock.eval.runner.

        The dynamics are deterministic, so `seed` changes nothing yet; it is accepted for the
        Gymnasium-style interface."""
        self.state = State(
            t=0,
            agents=tuple((aid, tuple(self.scenario.agents[aid])) for aid in sorted(self.scenario.agents)),
            objects=tuple((oid, tuple(self.grid.objects[oid])) for oid in sorted(self.grid.objects)),
        )
        return self._observations(), {}

    def step(self, actions: dict[str, Action]):
        """Returns obs, rewards, terminated, truncated, info (info["events"]: list[Event]).

        rewards, terminated and truncated are {agent_id: value}. An agent is terminated once it
        reaches the goal and truncated when the step limit runs out first; a finished agent stays
        on its goal with reward 0 while the others play on."""
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        self.state, rewards, events = self.simulate(self.state, actions)
        terminated = {aid: aid in self.state.done for aid, _ in self.state.agents}
        truncated = {aid: self.state.t >= self.max_steps and not done for aid, done in terminated.items()}
        return self._observations(), rewards, terminated, truncated, {"events": events}

    def observe(self, agent_id: str) -> Any:
        """This agent's view of the current state; see ObservationEncoder."""
        return self.encoder.encode(self.state, agent_id)

    # Lookahead hooks for the A* and CSP planners.
    def get_state(self) -> State:
        return self.state

    def set_state(self, state: State) -> None:
        self.state = state

    def legal_actions(self, state: State, agent_id: str) -> list[Action]:
        return self.referee.legal_actions(state, agent_id)

    def simulate(self, state: State, actions: dict[str, Action]) -> tuple[State, dict[str, float], list[Event]]:
        """Pure transition: no side effects on self.state. Actions may be Actions or Gymnasium
        action indices; agents left out stay put. Finished agents get reward 0."""
        actions = {aid: action if isinstance(action, Action) else ACTIONS[int(action)]
                   for aid, action in actions.items()}
        next_state, events = self.referee.resolve(state, actions, state.t + 1)
        rewards = {aid: 0.0 if aid in state.done else float(self.reward_fn(next_state, events, aid))
                   for aid, _ in next_state.agents}
        return next_state, rewards, events

    def _observations(self) -> dict[str, Any]:
        return {aid: self.observe(aid) for aid, _ in self.state.agents}

    def render(self, mode: str = "ascii", ax=None):
        """Draw the current state: "ascii" returns a string, "matplotlib" returns the axes."""
        if self.state is None:
            raise RuntimeError("call reset() before render()")
        agents = dict(self.state.agents)
        objects = object_positions(dict(self.state.objects))
        if mode == "ascii":
            return self.grid.render_ascii(agents, objects)
        if mode == "matplotlib":
            return self.grid.render_matplotlib(agents, ax=ax, objects=objects)
        raise ValueError(f"unknown render mode {mode!r}; use 'ascii' or 'matplotlib'")
