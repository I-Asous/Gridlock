"""R*: per-scenario intent predicates and score."""

from gridlock.env.events import Event


class IntentFn:
    def __init__(self, config: dict):
        self.config = config   # e.g. {"forbid": [{"kind": "BROKE", "target": "vase"}], ...}

    def step_score(self, state, events: list[Event], agent_id: str) -> float:
        raise NotImplementedError

    def violations(self, state, events: list[Event]) -> list[str]:
        """Names of intent predicates violated by these events."""
        raise NotImplementedError
