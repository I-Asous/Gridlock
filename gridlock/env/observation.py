"""What an agent observes: a fixed-length integer vector, which is also the Gymnasium observation space."""

from typing import Any, Iterable, Mapping, Optional

import numpy as np
from gymnasium import spaces

from gridlock.env.events import Pos


def object_positions(objects: Mapping[str, Any]) -> dict[str, Pos]:
    """Positions of objects still on the board. A value may be a position or a dict with a "pos"
    key; anything else (e.g. None once an object is picked up) is skipped."""
    positions = {}
    for oid, value in dict(objects).items():
        if isinstance(value, Mapping):
            value = value.get("pos")
        if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(c, int) for c in value):
            positions[oid] = (value[0], value[1])
    return positions


class ObservationEncoder:
    """Encodes a State from one agent's point of view as

        [own x, own y,  other agents' x, y (sorted by id),  objects' x, y (sorted by id),  one bit per flag]

    A position of (width, height) means "not on the board", e.g. a broken or picked-up object.
    The step counter is left out, so the same situation at different times looks the same; that
    keeps a Q-table from treating every timestep as a new state.
    """

    def __init__(self, width: int, height: int, agent_ids: Iterable[str], object_ids: Iterable[str] = (),
                 flag_names: Iterable[str] = ()):
        self.width, self.height = width, height
        self.agent_ids = sorted(agent_ids)
        self.object_ids = sorted(object_ids)
        self.flag_names = list(flag_names)
        n_positions = len(self.agent_ids) + len(self.object_ids)
        self.space = spaces.MultiDiscrete([width + 1, height + 1] * n_positions + [2] * len(self.flag_names),
                                          dtype=np.int64)

    @classmethod
    def for_scenario(cls, scenario) -> "ObservationEncoder":
        grid = scenario.grid
        return cls(grid.width, grid.height, scenario.agents, grid.objects, getattr(scenario, "flags", ()))

    def encode(self, state, agent_id: str) -> np.ndarray:
        if agent_id not in self.agent_ids:
            raise ValueError(f"unknown agent {agent_id!r}; expected one of {self.agent_ids}")
        agents = dict(state.agents)
        objects = object_positions(dict(state.objects))
        values = []
        for aid in [agent_id] + [aid for aid in self.agent_ids if aid != agent_id]:
            values.extend(self._position(agents.get(aid)))
        for oid in self.object_ids:
            values.extend(self._position(objects.get(oid)))
        values.extend(int(name in state.flags) for name in self.flag_names)
        return np.array(values, dtype=np.int64)

    def _position(self, pos: Optional[Pos]) -> tuple[int, int]:
        return (self.width, self.height) if pos is None else (int(pos[0]), int(pos[1]))
