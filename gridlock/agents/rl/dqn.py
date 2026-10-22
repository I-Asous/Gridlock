"""DQN fallback for scenarios too large for a Q-table. Implement only if needed."""

from gridlock.agents.base import Agent


class DQNAgent(Agent):
    def act(self, obs):
        raise NotImplementedError
