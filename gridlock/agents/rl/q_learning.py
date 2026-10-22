"""Tabular Q-learning on hashed environment states. Trained on R only."""

import random
from collections import defaultdict

from gridlock.agents.base import Agent
from gridlock.env.grid import Action


class QLearningAgent(Agent):
    def __init__(self, agent_id: str, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, seed: int | None = None):
        super().__init__(agent_id)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q: dict = defaultdict(lambda: {a: 0.0 for a in Action})
        self.training = True

    def act(self, obs) -> Action:
        raise NotImplementedError

    def update(self, obs, action: Action, reward: float, next_obs, done: bool) -> None:
        raise NotImplementedError

    def save(self, path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path) -> "QLearningAgent":
        raise NotImplementedError
