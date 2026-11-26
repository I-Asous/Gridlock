"""Static grid layout, tiles, objects and rendering."""

from collections import deque
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
    TRAP = "^"               # stepping on it sends the agent back to its start


# Board palette: drawn like a plan, black walls and hairline cell lines on white, so any agent
# colour stands out. Special tiles are told apart by flat gray, hatching and glyphs.
TILE_COLORS = {
    Tile.FLOOR: "#ffffff",
    Tile.WALL: "#000000",
    Tile.GOAL: "#a5a5a5",
    Tile.CHECKPOINT: "#e4e4e4",
    Tile.SWITCH: "#ffffff",
    Tile.TRAP: "#ffffff",
}
TILE_GLYPHS = {Tile.GOAL: ("⚑", "#000000"), Tile.CHECKPOINT: ("★", "#000000"), Tile.SWITCH: ("S", "#000000")}
BOARD_BG = "#000000"          # cell lines and outer frame
TRAP_STRIPES = "#000000"
OBJECT_COLOR = "#ffffff"      # objects: white squares ruled in black
OBJECT_EDGE = "#000000"
OBJECT_GLYPHS = {"vase": "⚱"}
# No red: the viewers in gridlock.viz reserve it for monitor flags.
AGENT_COLORS = ("#1f3bff", "#ff7a00", "#00a06a", "#7a2cff", "#c2187a", "#0091b8", "#6b7a00", "#8a5a00")


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

    def cells(self, tile: Tile) -> list[Pos]:
        """Every position holding `tile`, row by row."""
        return [(x, y) for y, row in enumerate(self.tiles) for x, t in enumerate(row) if t is tile]

    def distances_from(self, sources: Iterable[Pos], avoid: Iterable[Tile] = ()) -> dict[Pos, int]:
        """Steps from the nearest source to every reachable cell, moving up/down/left/right
        around walls and any `avoid` tiles (breadth-first search). Unreachable cells are left out."""
        avoid = set(avoid)
        distance: dict[Pos, int] = {}
        queue: deque[Pos] = deque()
        for pos in sources:
            if self.is_walkable(pos) and pos not in distance:
                distance[pos] = 0
                queue.append(pos)
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                nxt = (x + dx, y + dy)
                if nxt not in distance and self.is_walkable(nxt) and self.tile_at(nxt) not in avoid:
                    distance[nxt] = distance[(x, y)] + 1
                    queue.append(nxt)
        return distance

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
                          colors: Optional[dict[str, str]] = None, legend: bool = True):
        """Draw tiles, objects and agents on `ax` (a new figure when None) and return the axes.

        (0, 0) is the top-left cell and y grows downward, matching Action.DOWN = (0, 1).
        `colors` maps agent id -> colour and fixes each agent's number; pass the same mapping
        to every call so agents keep their colour when others leave the board.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        if ax is None:
            _, ax = plt.subplots(figsize=(max(3.0, self.width * 0.6), max(3.0, self.height * 0.6)))
        ax.clear()
        self.draw_board(ax)
        objects = self.objects if objects is None else objects
        colors = agent_palette(agents) if colors is None else colors
        self.draw_pieces(ax, agents, objects, colors)
        if legend:
            labels = {aid: _agent_glyph(i) for i, aid in enumerate(sorted(colors))}
            handles = [Line2D([], [], ls="", marker="o", ms=9, mfc=colors.get(aid, "#6b7280"), mec="#000000",
                              label=f"{labels.get(aid, '?')}  {aid}") for aid in sorted(agents)]
            handles += [Line2D([], [], ls="", marker="s", ms=8, mfc=OBJECT_COLOR, mec=OBJECT_EDGE, label=oid)
                        for oid in sorted(objects)]
            if handles:
                ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                          ncol=min(len(handles), 4), frameon=False, fontsize=8)
        return ax

    def draw_board(self, ax) -> None:
        """The static layer: square tiles split by hairlines inside a black frame, with glyphs for
        special tiles. Draw it once and then only redraw pieces (see draw_pieces)."""
        from matplotlib.patches import Rectangle

        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        size = _glyph_size(ax, self)
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc=TILE_COLORS[tile], ec=BOARD_BG, lw=0.5, zorder=1))
                if tile is Tile.TRAP:
                    stripes = Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, hatch="////", lw=0, zorder=1.2)
                    stripes.set_edgecolor(TRAP_STRIPES)
                    ax.add_patch(stripes)
                elif tile in TILE_GLYPHS:
                    glyph, color = TILE_GLYPHS[tile]
                    ax.text(x, y + 0.02, glyph, ha="center", va="center", fontsize=size, color=color, zorder=1.5)
        ax.add_patch(Rectangle((-0.5, -0.5), self.width, self.height, fill=False, ec=BOARD_BG, lw=1.5, zorder=1.6))

    def draw_pieces(self, ax, agents: dict[str, Pos], objects: dict[str, Pos], colors: dict[str, str]) -> list:
        """The moving layer: objects and agent tokens. Returns the artists, so a caller that keeps
        the board can remove just these before the next frame."""
        from matplotlib.patches import Circle, Rectangle

        artists = []
        size = _glyph_size(ax, self)
        labels = {aid: _agent_glyph(i) for i, aid in enumerate(sorted(colors))}
        for oid, (x, y) in objects.items():
            artists.append(ax.add_patch(Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, fc=OBJECT_COLOR, ec=OBJECT_EDGE,
                                                  lw=1.2, zorder=2)))
            artists.append(ax.text(x, y + 0.02, OBJECT_GLYPHS.get(oid, oid[0].lower()), ha="center", va="center",
                                   fontsize=size * 0.9, color=OBJECT_EDGE, zorder=2.5))
        sharing: dict[Pos, list[str]] = {}
        for aid in sorted(agents):
            sharing.setdefault(tuple(agents[aid]), []).append(aid)
        for (x, y), ids in sharing.items():
            # Agents on the same cell sit side by side, shrunk to fit.
            radius = 0.36 if len(ids) == 1 else 0.4 / len(ids)
            for i, aid in enumerate(ids):
                cx = x + (i - (len(ids) - 1) / 2) * 0.8 / len(ids)
                artists.append(ax.add_patch(Circle((cx, y), radius, fc=colors.get(aid, "#6b7280"), ec="#000000",
                                                   lw=1.2, zorder=3)))
                artists.append(ax.text(cx, y + 0.01, labels.get(aid, "?"), ha="center", va="center",
                                       fontsize=size * (1 if len(ids) == 1 else 0.75), color="white", family="sans-serif",
                                       zorder=3.5))
        return artists


def _glyph_size(ax, grid: "Grid") -> float:
    """A font size (points) that fits about half a cell, whatever the board's size on screen."""
    box = ax.get_window_extent()
    cell_pts = min(box.width / grid.width, box.height / grid.height) * 72 / ax.figure.dpi
    return max(6.0, min(16.0, cell_pts * 0.48))
