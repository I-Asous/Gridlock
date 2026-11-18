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


# Board palette: pale special tiles so any agent colour stands out on them.
TILE_COLORS = {
    Tile.FLOOR: "#ffffff",
    Tile.WALL: "#334155",
    Tile.GOAL: "#d1fae5",
    Tile.CHECKPOINT: "#fef3c7",
    Tile.SWITCH: "#ede9fe",
    Tile.TRAP: "#facc15",
}
TILE_GLYPHS = {Tile.GOAL: ("⚑", "#047857"), Tile.CHECKPOINT: ("★", "#d97706"), Tile.SWITCH: ("S", "#7c3aed")}
BOARD_BG = "#e2e8f0"          # gutter between tiles
TRAP_STRIPES = "#1f2937"
OBJECT_COLOR = "#92400e"
OBJECT_GLYPHS = {"vase": "⚱"}
# No red: the viewers in gridlock.viz reserve it for monitor flags.
AGENT_COLORS = ("#2563eb", "#ea580c", "#059669", "#7c3aed", "#db2777", "#0891b2", "#65a30d", "#a16207")


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
            handles = [Line2D([], [], ls="", marker="o", ms=9, mfc=colors.get(aid, "#6b7280"), mec="white",
                              label=f"{labels.get(aid, '?')}  {aid}") for aid in sorted(agents)]
            handles += [Line2D([], [], ls="", marker="s", ms=8, mfc=OBJECT_COLOR, mec="white", label=oid)
                        for oid in sorted(objects)]
            if handles:
                ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                          ncol=min(len(handles), 4), frameon=False, fontsize=8)
        return ax

    def draw_board(self, ax) -> None:
        """The static layer: rounded tiles on a light gutter, with glyphs for special tiles.
        Draw it once and then only redraw pieces (see draw_pieces)."""
        from matplotlib.patches import FancyBboxPatch

        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(FancyBboxPatch((-0.5, -0.5), self.width, self.height, boxstyle="round,pad=0,rounding_size=0.3",
                                    fc=BOARD_BG, ec="none", zorder=0))
        gap, round_ = 0.04, 0.14
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                box = dict(boxstyle=f"round,pad=0,rounding_size={round_}", ec="none", zorder=1)
                ax.add_patch(FancyBboxPatch((x - 0.5 + gap, y - 0.5 + gap), 1 - 2 * gap, 1 - 2 * gap,
                                            fc=TILE_COLORS[tile], **box))
                if tile is Tile.TRAP:
                    stripes = FancyBboxPatch((x - 0.5 + gap, y - 0.5 + gap), 1 - 2 * gap, 1 - 2 * gap, fill=False,
                                             hatch="////", lw=0, **{**box, "zorder": 1.2})
                    stripes.set_edgecolor(TRAP_STRIPES)
                    ax.add_patch(stripes)
                elif tile in TILE_GLYPHS:
                    glyph, color = TILE_GLYPHS[tile]
                    ax.text(x, y + 0.02, glyph, ha="center", va="center", fontsize=_glyph_size(ax, self), color=color,
                            fontweight="bold", zorder=1.5)

    def draw_pieces(self, ax, agents: dict[str, Pos], objects: dict[str, Pos], colors: dict[str, str]) -> list:
        """The moving layer: objects and agent tokens. Returns the artists, so a caller that keeps
        the board can remove just these before the next frame."""
        from matplotlib.patches import Circle, FancyBboxPatch

        artists = []
        size = _glyph_size(ax, self)
        labels = {aid: _agent_glyph(i) for i, aid in enumerate(sorted(colors))}
        for oid, (x, y) in objects.items():
            artists.append(ax.add_patch(FancyBboxPatch((x - 0.3, y - 0.3), 0.6, 0.6, boxstyle="round,pad=0,rounding_size=0.12",
                                                       fc=OBJECT_COLOR, ec="white", lw=1.2, zorder=2)))
            artists.append(ax.text(x, y + 0.02, OBJECT_GLYPHS.get(oid, oid[0].lower()), ha="center", va="center",
                                   fontsize=size, color="white", fontweight="bold", zorder=2.5))
        sharing: dict[Pos, list[str]] = {}
        for aid in sorted(agents):
            sharing.setdefault(tuple(agents[aid]), []).append(aid)
        for (x, y), ids in sharing.items():
            # Agents on the same cell sit side by side, shrunk to fit.
            radius = 0.36 if len(ids) == 1 else 0.4 / len(ids)
            for i, aid in enumerate(ids):
                cx = x + (i - (len(ids) - 1) / 2) * 0.8 / len(ids)
                artists.append(ax.add_patch(Circle((cx + 0.04, y + 0.06), radius, fc="#0f172a", ec="none", alpha=0.18,
                                                   zorder=2.9)))
                artists.append(ax.add_patch(Circle((cx, y), radius, fc=colors.get(aid, "#6b7280"), ec="white", lw=2,
                                                   zorder=3)))
                artists.append(ax.text(cx, y + 0.01, labels.get(aid, "?"), ha="center", va="center",
                                       fontsize=size * (1 if len(ids) == 1 else 0.75), fontweight="bold",
                                       color="white", zorder=3.5))
        return artists


def _glyph_size(ax, grid: "Grid") -> float:
    """A font size (points) that fits about half a cell, whatever the board's size on screen."""
    box = ax.get_window_extent()
    cell_pts = min(box.width / grid.width, box.height / grid.height) * 72 / ax.figure.dpi
    return max(6.0, min(16.0, cell_pts * 0.48))
