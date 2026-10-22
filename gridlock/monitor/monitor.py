"""Online R*-aware monitor: consumes events step by step and flags intent divergence."""

from dataclasses import dataclass

from gridlock.env.events import Event
from gridlock.intent.intent import IntentFn


@dataclass(frozen=True)
class Flag:
    t: int
    agent_id: str
    reason: str


class Monitor:
    def __init__(self, intent: IntentFn, divergence_threshold: float):
        self.intent = intent
        self.threshold = divergence_threshold
        self.flags: list[Flag] = []

    def reset(self) -> None:
        self.flags = []

    def observe(self, t: int, state, events: list[Event], rewards: dict[str, float]) -> list[Flag]:
        """Process one step; return any new flags."""
        raise NotImplementedError
