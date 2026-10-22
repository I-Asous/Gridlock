"""Static grid layout, tiles, objects and rendering."""

from dataclasses import dataclass, field
from enum import Enum

from gridlock.env.events import Pos


class Action(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    STAY = (0, 0)
    INTERACT = None          # toggle / pick up on current or facing tile


class Tile(Enum):
    FLOOR = "."
    WALL = "#"
    GOAL = "G"
    CHECKPOINT = "C"
    SWITCH = "S"


@dataclass
class Grid:
    width: int
    height: int
    tiles: list[list[Tile]]
    objects: dict[str, Pos] = field(default_factory=dict)   # e.g. {"vase": (3, 2)}

    def in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def tile_at(self, pos: Pos) -> Tile:
        x, y = pos
        return self.tiles[y][x]

    def is_walkable(self, pos: Pos) -> bool:
        return self.in_bounds(pos) and self.tile_at(pos) is not Tile.WALL

    def render_ascii(self, agents: dict[str, Pos]) -> str:
        raise NotImplementedError

    def render_matplotlib(self, agents: dict[str, Pos], ax=None):
        raise NotImplementedError
