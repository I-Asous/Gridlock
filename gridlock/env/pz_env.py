"""PettingZoo view of a GridWorld: every agent is controlled from outside, all acting at once.

GridlockEnv (gym_env.py) gives one learner a single-agent view with the other agents as fixed
policies. This is the multi-agent counterpart: a PettingZoo ParallelEnv, so several agents can
learn at the same time, with gridlock's own independent learners (gridlock.agents.rl.multi) or
with any PettingZoo-compatible library.

    from gridlock.env.pz_env import parallel_env
    env = parallel_env("commons")
    observations, infos = env.reset(seed=0)
    while env.agents:
        actions = {aid: env.action_space(aid).sample() for aid in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

Each agent observes the usual vector (gridlock.env.observation) and is rewarded with its own R.
R* is never exposed. Every live agent's info carries the step's events. Following PettingZoo, an
agent leaves `env.agents` once it reaches the goal (terminated) or the step limit runs out
(truncated); the underlying GridWorld keeps it on its goal, where it no longer blocks anyone.

PettingZoo is optional (pip install pettingzoo); only this module needs it.
"""

from typing import Optional

from gymnasium import spaces
from pettingzoo import ParallelEnv

from gridlock.env.gridworld import GridWorld
from gridlock.env.gym_env import ACTIONS, to_action
from gridlock.env.observation import object_positions


class GridlockParallelEnv(ParallelEnv):
    metadata = {"name": "gridlock_v0", "render_modes": ["ansi"], "is_parallelizable": True}

    def __init__(self, world: GridWorld, render_mode: Optional[str] = None):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.world = world
        self.render_mode = render_mode
        self.possible_agents = sorted(world.scenario.agents)
        self.agents: list[str] = []
        self._observation_space = world.encoder.space
        self._action_space = spaces.Discrete(len(ACTIONS))

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._action_space.seed(seed)
        observations, _ = self.world.reset(seed=seed)
        self.agents = list(self.possible_agents)
        return {aid: observations[aid] for aid in self.agents}, {aid: {} for aid in self.agents}

    def step(self, actions: dict):
        """`actions` maps each live agent to a Gymnasium action index or an Action. Live agents
        left out stay put. Returns dicts keyed by the agents that were live at this step."""
        live = list(self.agents)
        observations, rewards, terminated, truncated, info = self.world.step(
            {aid: to_action(actions[aid]) for aid in live if aid in actions})
        events = info["events"]
        self.agents = [aid for aid in live if not (terminated[aid] or truncated[aid])]
        return ({aid: observations[aid] for aid in live},
                {aid: float(rewards[aid]) for aid in live},
                {aid: bool(terminated[aid]) for aid in live},
                {aid: bool(truncated[aid]) for aid in live},
                {aid: {"events": events} for aid in live})

    def render(self):
        if self.render_mode == "ansi":
            state = self.world.get_state()
            return self.world.grid.render_ascii(dict(state.agents), object_positions(dict(state.objects)))
        return None

    def close(self) -> None:
        pass


def parallel_env(scenario: str, max_steps: Optional[int] = None, render_mode: Optional[str] = None) -> GridlockParallelEnv:
    """A GridlockParallelEnv for a level by name. max_steps defaults to the level's own, or 100."""
    from gridlock.scenarios import load_scenario

    level = load_scenario(scenario)
    return GridlockParallelEnv(GridWorld(level, max_steps=max_steps or level.max_steps or 100), render_mode)
