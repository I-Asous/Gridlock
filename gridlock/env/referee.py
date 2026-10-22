"""Rule referee: legality checks, move application, multi-agent conflict resolution."""

from gridlock.env.events import Event
from gridlock.env.grid import Action, Grid


class Referee:
    def __init__(self, grid: Grid):
        self.grid = grid

    def legal_actions(self, state, agent_id: str) -> list[Action]:
        raise NotImplementedError

    def resolve(self, state, actions: dict[str, Action], t: int) -> tuple[object, list[Event]]:
        """Apply joint actions simultaneously; return (next_state, events)."""
        raise NotImplementedError
