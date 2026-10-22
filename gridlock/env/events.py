"""Typed event log. Every behaviour relevant to R* must surface as an Event."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

Pos = tuple[int, int]


class EventKind(Enum):
    MOVED = auto()
    BUMPED = auto()          # tried to move into a wall/occupied cell
    PICKED_UP = auto()
    PUSHED = auto()
    BROKE = auto()
    TOGGLED = auto()         # switches, sensors, off-buttons
    BLOCKED = auto()         # agent occupies a cell another agent wanted
    CHECKPOINT = auto()
    REACHED_GOAL = auto()


@dataclass(frozen=True)
class Event:
    t: int
    agent_id: str
    kind: EventKind
    pos: Pos
    target: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
