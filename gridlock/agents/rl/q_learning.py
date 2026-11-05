"""Tabular Q-learning over Gymnasium observations. Trained on R only.

Q-values are kept per observation as a numpy row with one entry per Gymnasium action index
(gridlock.env.gym_env.ACTIONS).
"""

import pickle
from collections import defaultdict
from functools import partial
from typing import Hashable, Optional

import numpy as np

from gridlock.agents.base import Agent
from gridlock.env.grid import Action
from gridlock.env.gym_env import ACTIONS


def state_key(obs) -> Hashable:
    """Q-table key. Gymnasium observations are numpy arrays, which can't be dict keys."""
    return tuple(obs.tolist()) if isinstance(obs, np.ndarray) else obs


def action_index(action) -> int:
    return ACTIONS.index(action) if isinstance(action, Action) else int(action)


class QLearningAgent(Agent):
    def __init__(self, agent_id: str, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.995, seed: Optional[int] = None):
        super().__init__(agent_id)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q: defaultdict = defaultdict(partial(np.zeros, len(ACTIONS)))
        self.training = True

    def act(self, obs) -> Action:
        """ε-greedy while training, greedy otherwise; set `training = False` to evaluate.
        GridlockEnv.step accepts the returned Action as well as an action index."""
        if self.training and self.rng.random() < self.epsilon:
            return ACTIONS[self.rng.integers(len(ACTIONS))]
        return self.greedy(obs)

    def greedy(self, obs) -> Action:
        """Highest-valued action, ties broken at random. Looking up an unseen state doesn't add it."""
        row = self.q.get(state_key(obs))
        if row is None:
            return ACTIONS[self.rng.integers(len(ACTIONS))]
        return ACTIONS[self.rng.choice(np.flatnonzero(row == row.max()))]

    def update(self, obs, action, reward: float, next_obs, terminated: bool) -> None:
        """One Q-learning backup. Pass Gymnasium's `terminated`, not `truncated`: when an episode
        is only cut off by the time limit, the target should still bootstrap from next_obs."""
        row = self.q[state_key(obs)]
        next_row = self.q.get(state_key(next_obs))
        future = 0.0 if terminated or next_row is None else float(next_row.max())
        i = action_index(action)
        row[i] += self.alpha * (reward + self.gamma * future - row[i])

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path) -> None:
        params = {name: getattr(self, name)
                  for name in ("agent_id", "alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay")}
        with open(path, "wb") as f:
            pickle.dump({"params": params, "q": dict(self.q)}, f)

    @classmethod
    def load(cls, path, seed: Optional[int] = None) -> "QLearningAgent":
        """Restore a checkpoint written by save(). It is a pickle, so only load files you trust."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(**data["params"], seed=seed)
        agent.q.update(data["q"])
        return agent
