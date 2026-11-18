"""Gymnasium view of a GridWorld for one learning agent.

The other agents are fixed policies that act inside step(), so the learner sees an ordinary
single-agent Gymnasium environment and any Gymnasium-compatible algorithm can train on it.
Only R comes out of step(); R* stays in gridlock.intent.

    import gymnasium
    import gridlock.env   # registers the environment
    env = gymnasium.make("gridlock/Gridlock-v0", scenario="vase")
"""

from typing import Any, Optional

import gymnasium
from gymnasium import spaces

from gridlock.env.grid import Action
from gridlock.env.observation import ObservationEncoder, object_positions

ACTIONS = tuple(Action)          # Gymnasium action i is ACTIONS[i]
ENV_ID = "gridlock/Gridlock-v0"


def to_action(action) -> Action:
    """Accept a Gymnasium action index or an Action."""
    return action if isinstance(action, Action) else ACTIONS[int(action)]


class GridlockEnv(gymnasium.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, world, learner_id: str, opponents: Optional[dict] = None,
                 render_mode: Optional[str] = None):
        """`world` is a GridWorld, or anything with its reset/step/get_state interface and a
        `scenario`. `opponents` maps other agent ids to Agent objects; an agent without one stays put."""
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.world = world
        self.learner_id = learner_id
        self.opponents = dict(opponents or {})
        self.other_ids = [aid for aid in sorted(world.scenario.agents) if aid != learner_id]
        self.encoder = ObservationEncoder.for_scenario(world.scenario)
        self.observation_space = self.encoder.space
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.render_mode = render_mode
        self._world_obs: dict[str, Any] = {}

    @property
    def state(self):
        """The underlying GridWorld State, for analysis code such as R* logging. Not an observation."""
        return self.world.get_state()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Derive the world's seed from np_random so a whole run is reproducible from the first seed.
        self._world_obs, info = self.world.reset(seed=int(self.np_random.integers(2**31 - 1)))
        for agent in self.opponents.values():
            agent.reset(self.world)
        return self._observe(), dict(info)

    def step(self, action):
        actions = {self.learner_id: to_action(action)}
        for aid in self.other_ids:
            policy = self.opponents.get(aid)
            actions[aid] = policy.act(self._world_obs[aid]) if policy is not None else Action.STAY
        self._world_obs, rewards, terminated, truncated, info = self.world.step(actions)
        lid = self.learner_id
        return (self._observe(), float(rewards[lid]), _for_agent(terminated, lid), _for_agent(truncated, lid),
                dict(info))

    def render(self):
        if self.render_mode == "ansi":
            state = self.world.get_state()
            return self.world.scenario.grid.render_ascii(dict(state.agents), object_positions(dict(state.objects)))
        return None

    def _observe(self):
        return self.encoder.encode(self.world.get_state(), self.learner_id)


def _for_agent(flag, agent_id: str) -> bool:
    """GridWorld done flags may be a bool or a per-agent dict."""
    return bool(flag[agent_id]) if isinstance(flag, dict) else bool(flag)


def make_gridlock_env(scenario: str, learner_id: Optional[str] = None, max_steps: int = 100,
                      seed: Optional[int] = None, render_mode: Optional[str] = None) -> GridlockEnv:
    """Build the environment for a level. The learner is `learner_id` (default: the first agent
    by sorted id) and every other agent acts at random."""
    from gridlock.agents.scripted import RandomAgent
    from gridlock.env.gridworld import GridWorld
    from gridlock.scenarios import load_scenario

    level = load_scenario(scenario)
    agent_ids = sorted(level.agents)
    learner_id = learner_id or agent_ids[0]
    others = [aid for aid in agent_ids if aid != learner_id]
    opponents = {aid: RandomAgent(aid, seed=None if seed is None else seed + i + 1) for i, aid in enumerate(others)}
    return GridlockEnv(GridWorld(level, max_steps=max_steps), learner_id, opponents, render_mode=render_mode)


if ENV_ID not in gymnasium.registry:
    gymnasium.register(id=ENV_ID, entry_point=make_gridlock_env)
