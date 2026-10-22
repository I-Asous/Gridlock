"""Common agent interface."""

from abc import ABC, abstractmethod
from typing import Any

from gridlock.env.grid import Action


class Agent(ABC):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def reset(self, env) -> None:
        """Called at episode start; planners compute their plan here."""

    @abstractmethod
    def act(self, obs: Any) -> Action:
        ...
