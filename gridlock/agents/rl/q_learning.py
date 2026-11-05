"""Tabular Q-learning on hashed environment states. Trained on R only."""

import pickle
import random
from collections import defaultdict
from typing import Hashable, Optional

from gridlock.agents.base import Agent
from gridlock.env.grid import Action
from gridlock.env.gridworld import State

ACTIONS = tuple(Action)


def _zero_row() -> dict[Action, float]:
    return {action: 0.0 for action in ACTIONS}


def state_key(obs) -> Hashable:
    """Q-table key for an observation.

    A State's step counter is dropped: with it, every timestep would be a separate table entry
    and nothing learned at t=5 would carry over to t=6. Any other observation must be hashable.
    """
    if isinstance(obs, State):
        return obs.agents, obs.objects, obs.flags
    return obs


class QLearningAgent(Agent):
    def __init__(self, agent_id: str, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.995, seed: Optional[int] = None):
        super().__init__(agent_id)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q: defaultdict = defaultdict(_zero_row)
        self.training = True

    def act(self, obs) -> Action:
        """ε-greedy while training, greedy otherwise. Set `training = False` to evaluate."""
        if self.training and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        return self.greedy(obs)

    def greedy(self, obs) -> Action:
        """Highest-valued action, ties broken at random. Looking up an unseen state doesn't add it."""
        row = self.q.get(state_key(obs))
        if row is None:
            return self.rng.choice(ACTIONS)
        best = max(row.values())
        return self.rng.choice([action for action, value in row.items() if value == best])

    def update(self, obs, action: Action, reward: float, next_obs, done: bool) -> None:
        """One Q-learning backup. Pass done=True only when the episode really ended for this agent
        (terminated); on a time-limit truncation pass False so the target still bootstraps."""
        row = self.q[state_key(obs)]
        target = reward if done else reward + self.gamma * max(self.q[state_key(next_obs)].values())
        row[action] += self.alpha * (target - row[action])

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
