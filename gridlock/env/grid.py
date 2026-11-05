"""Static grid layout, tiles, objects and rendering."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

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


TILE_COLORS = {
    Tile.FLOOR: "#f5f3ee",
    Tile.WALL: "#3f3f46",
    Tile.GOAL: "#86efac",
    Tile.CHECKPOINT: "#fde68a",
    Tile.SWITCH: "#c4b5fd",
}
OBJECT_COLOR = "#a16207"
# No red: the viewers in gridlock.viz reserve it for monitor flags.
AGENT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22")


def agent_palette(agent_ids: Iterable[str]) -> dict[str, str]:
    """Agent id -> colour, assigned in sorted-id order so every view agrees."""
    return {aid: AGENT_COLORS[i % len(AGENT_COLORS)] for i, aid in enumerate(sorted(agent_ids))}


def _agent_glyph(index: int) -> str:
    return str(index) if index < 10 else "@"


@dataclass
class Grid:
    width: int
    height: int
    tiles: list[list[Tile]]
    objects: dict[str, Pos] = field(default_factory=dict)   # e.g. {"vase": (3, 2)}

    @classmethod
    def from_rows(cls, rows: list[str], objects: Optional[dict[str, Pos]] = None) -> "Grid":
        """Build a grid from rows of tile characters, e.g. ["#####", "#..G#", "#####"]."""
        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError("rows must be non-empty and all the same width")
        tiles = [[Tile(ch) for ch in row] for row in rows]
        return cls(width=len(rows[0]), height=len(rows), tiles=tiles, objects=dict(objects or {}))

    def to_rows(self) -> list[str]:
        return ["".join(tile.value for tile in row) for row in self.tiles]

    def in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def tile_at(self, pos: Pos) -> Tile:
        x, y = pos
        return self.tiles[y][x]

    def is_walkable(self, pos: Pos) -> bool:
        return self.in_bounds(pos) and self.tile_at(pos) is not Tile.WALL

    def render_ascii(self, agents: dict[str, Pos], objects: Optional[dict[str, Pos]] = None) -> str:
        """Tiles as their characters, objects as the lowercase first letter of their id, and
        agents as digits in sorted-id order. A legend line follows when anything is placed.
        `objects` defaults to the layout's starting positions."""
        objects = self.objects if objects is None else objects
        cells = [[tile.value for tile in row] for row in self.tiles]
        object_legend = []
        for oid in sorted(objects):
            x, y = objects[oid]
            cells[y][x] = oid[0].lower()
            object_legend.append(f"{oid[0].lower()}={oid}")
        agent_legend = []
        for i, aid in enumerate(sorted(agents)):
            x, y = agents[aid]
            cells[y][x] = _agent_glyph(i)
            agent_legend.append(f"{_agent_glyph(i)}={aid}")
        text = "\n".join("".join(row) for row in cells)
        legend = agent_legend + object_legend
        return text + "\n" + "  ".join(legend) if legend else text

    def render_matplotlib(self, agents: dict[str, Pos], ax=None, objects: Optional[dict[str, Pos]] = None,
                          colors: Optional[dict[str, str]] = None):
        """Draw tiles, objects and agents on `ax` (a new figure when None) and return the axes.

        (0, 0) is the top-left cell and y grows downward, matching Action.DOWN = (0, 1).
        `colors` maps agent id -> colour and fixes each agent's number; pass the same mapping
        to every call so agents keep their colour when others leave the board.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Rectangle

        if ax is None:
            _, ax = plt.subplots(figsize=(max(3.0, self.width * 0.6), max(3.0, self.height * 0.6)))
        objects = self.objects if objects is None else objects
        colors = agent_palette(agents) if colors is None else colors
        labels = {aid: _agent_glyph(i) for i, aid in enumerate(sorted(colors))}

        order = list(Tile)
        ax.imshow([[order.index(tile) for tile in row] for row in self.tiles],
                  cmap=ListedColormap([TILE_COLORS[tile] for tile in order]),
                  vmin=0, vmax=len(order) - 1, interpolation="nearest")
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if tile in (Tile.GOAL, Tile.CHECKPOINT, Tile.SWITCH):
                    ax.text(x, y, tile.value, ha="center", va="center", fontsize=9,
                            fontweight="bold", color="#00000060")

        for oid, (x, y) in objects.items():
            ax.add_patch(Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, fc=OBJECT_COLOR, ec="white", lw=1, zorder=2))
            ax.text(x, y, oid[0].lower(), ha="center", va="center", fontsize=9, color="white", zorder=2.5)

        for aid, (x, y) in agents.items():
            ax.add_patch(Circle((x, y), 0.36, fc=colors.get(aid, "#6b7280"), ec="white", lw=1.5, zorder=3))
            ax.text(x, y, labels.get(aid, "?"), ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=3.5)

        handles = [Line2D([], [], ls="", marker="o", ms=9, mfc=colors.get(aid, "#6b7280"), mec="white",
                          label=f"{labels.get(aid, '?')}  {aid}") for aid in sorted(agents)]
        handles += [Line2D([], [], ls="", marker="s", ms=8, mfc=OBJECT_COLOR, mec="white", label=oid)
                    for oid in sorted(objects)]
        if handles:
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                      ncol=min(len(handles), 4), frameon=False, fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks([x - 0.5 for x in range(self.width + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(self.height + 1)], minor=True)
        ax.grid(which="minor", color="#d4d4d8", lw=0.6)
        ax.tick_params(which="both", length=0)
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_aspect("equal")
        return ax
