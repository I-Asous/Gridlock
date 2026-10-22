"""R: the stated reward agents optimise. Configured per scenario."""

from gridlock.env.events import Event


class RewardFn:
    def __init__(self, config: dict):
        self.config = config   # e.g. {"step": -1, "goal": 10, "checkpoint": 1}

    def __call__(self, state, events: list[Event], agent_id: str) -> float:
        raise NotImplementedError
