"""The opening screen: every level, so the user can choose which one to view.

One card per level shows its starting board, its description and what the results file holds
for it (agents, episodes, seeds). Clicking a card opens the comparison viewer on that level's
first seed, and clicking one of its seed chips opens that seed instead. Keys 1-9 open the
first nine levels. From the comparison, "← levels" (or esc) comes back here.

Levels that ship in gridlock/scenarios/levels but have no episodes in the file are shown greyed
out, so it is clear they exist but have nothing to replay yet.

python3 -m gridlock.viz runs/sarsa_vs_q/results.jsonl
"""

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from gridlock.env.grid import Grid, agent_palette
from gridlock.viz import theme
from gridlock.viz.panels import draw_text_lines
from gridlock.viz.selection import level_entries, pick_comparison
from gridlock.viz.theme import GRAY, INK, MUTED, PAPER, Layout, themed

MAX_COLUMNS = 4
# Inches. The window is 14 x 8 like the other viewers and grows taller for more than one row.
MARGIN, GAP, HEADER_H, MIN_CARD_H = 0.25, 0.18, 1.12, 3.6
TEXT_H, SEEDS_H = 1.05, 0.42


def figure_size(n: int) -> tuple[float, float]:
    cols = max(1, min(n, MAX_COLUMNS))
    rows = -(-max(n, 1) // cols)
    return 14.0, max(8.0, HEADER_H + rows * MIN_CARD_H + (rows - 1) * GAP + MARGIN)


class LevelPicker:
    """A window with one card per level. Choosing one opens a ComparisonViewer for it and closes
    this window; that viewer's "← levels" button brings the picker back."""

    def __init__(self, trajectories, source: str = "", include_shipped: bool = True, interval_ms: int = 300):
        if not trajectories:
            raise ValueError("no episodes to choose from")
        self.trajectories = list(trajectories)
        self.source = source
        self.interval_ms = interval_ms
        self.levels = level_entries(self.trajectories, include_shipped)
        self.viewer = None
        self._open_window()

    # --- window ----------------------------------------------------------------------------------

    def _open_window(self) -> None:
        with themed():
            self.fig = plt.figure(figsize=figure_size(len(self.levels)))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Gridlock levels")
        self.fig._gridlock_owner = self               # callbacks hold only weak refs to bound methods
        self.hover: Optional[int] = None
        self.hover_chip = None
        self._build()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._size = tuple(self.fig.get_size_inches())
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)

    def _build(self) -> None:
        with themed():
            L = self.layout = Layout(self.fig)
            n = len(self.levels)
            recorded = sum(entry["recorded"] for entry in self.levels)
            L.title(MARGIN - 0.03, 0.02, "choose a level")
            source = f"{self.source}  /  " if self.source else ""
            L.text(MARGIN, 0.66, f"{source}{len(self.trajectories)} episodes  /  {recorded} of {n} levels recorded",
                   fontsize=8, color=INK)
            L.text(L.w - MARGIN, 0.66, "click a level to compare its agents   /   click a seed to open that seed"
                                       "   /   1–9  open", fontsize=7.5, color=MUTED, ha="right")
            L.rule(MARGIN, 0.92, L.w - 2 * MARGIN)
            L.edge_label("gridlock  /  levels")

            cols = max(1, min(n, MAX_COLUMNS))
            rows = -(-n // cols)
            card_w = (L.w - 2 * MARGIN - (cols - 1) * GAP) / cols
            card_h = (L.h - HEADER_H - MARGIN - (rows - 1) * GAP) / rows
            self.cards = []
            for k, entry in enumerate(self.levels):
                r, c = divmod(k, cols)
                self.cards.append(self._card(k, entry, MARGIN + c * (card_w + GAP), HEADER_H + r * (card_h + GAP),
                                             card_w, card_h))

    def _card(self, k: int, entry: dict, x: float, y: float, w: float, h: float) -> dict:
        L, live = self.layout, entry["recorded"]
        patch = L.card(x, y, w, h)[0]
        L.bar(x, y, w, 0.1, INK if live else GRAY)
        L.text(x + 0.14, y + 0.2, f"{k + 1:02d}", fontsize=7.5, color=MUTED)
        L.title(x + 0.45, y + 0.12, entry["name"], size=17, color=INK if live else MUTED)
        badge = f"{entry['episodes']} episode{'s' if entry['episodes'] != 1 else ''}" if live else "NOT RECORDED"
        L.text(x + w - 0.16, y + 0.3, badge, fontsize=7, color=INK, ha="right", va="center",
               bbox=theme.box(PAPER if live else GRAY))
        grid = Grid.from_rows(entry["grid_rows"])
        L.text(x + 0.48, y + 0.5, f"{grid.width} × {grid.height}  /  {len(entry['agents'])} agent"
                                  f"{'s' if len(entry['agents']) != 1 else ''} on the board", fontsize=7, color=MUTED)
        L.rule(x, y + 0.74, w, lw=theme.HAIR_W)

        board_h = h - 0.86 - TEXT_H - SEEDS_H - 0.2
        ax = L.axes(x + 0.14, y + 0.86, w - 0.28, board_h)
        grid.draw_board(ax)
        agents = {aid: tuple(pos) for aid, pos in entry["agents"].items()}
        grid.draw_pieces(ax, agents, {oid: tuple(pos) for oid, pos in entry["objects"].items()},
                         agent_palette(agents))
        if not live:
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=PAPER, ec="none", alpha=0.55, zorder=10))

        text_y = y + 0.86 + board_h + 0.08
        L.rule(x, text_y, w, lw=theme.HAIR_W)
        text_ax = L.axes(x + 0.14, text_y + 0.1, w - 0.28, TEXT_H - 0.1)
        draw_text_lines(self.fig, text_ax, self._card_lines(entry), wrap=True, fontsize=7.3, line_pts=10)

        chips = []
        if live:
            chips = self._seed_chips(entry, x + 0.14, y + h - SEEDS_H / 2 - 0.06, w - 0.28)
        return {"rect": L.rect(x, y, w, h), "patch": patch, "name": entry["name"], "recorded": live, "chips": chips}

    @staticmethod
    def _card_lines(entry: dict) -> list[tuple]:
        lines = [(entry["description"] or "No description.", INK if entry["recorded"] else MUTED)]
        if entry["recorded"]:
            lines.append((f"agents  {', '.join(entry['agent_names'])}", MUTED))
            lines.append((f"loophole in {entry['loophole_episodes']} of {entry['episodes']}   /   "
                          f"flagged in {entry['flagged_episodes']} of {entry['episodes']}", MUTED))
        else:
            lines.append(("No episodes in this file. Add the level to an experiment's scenarios to record it.",
                          MUTED))
        return lines

    def _seed_chips(self, entry: dict, x: float, y: float, w: float) -> list:
        """Boxed seed numbers in a row, as many as fit, then "+N more"."""
        L, renderer = self.layout, self.fig.canvas.get_renderer()
        label = L.text(x, y, "seeds", fontsize=7.5, color=MUTED, va="center")
        cursor = x + label.get_window_extent(renderer).width / self.fig.dpi + 0.2
        chips, seeds = [], entry["seeds"]
        for j, seed in enumerate(seeds):
            chip = L.text(cursor, y, f"{seed}", fontsize=7.5, color=INK, va="center", bbox=theme.box(PAPER))
            width = chip.get_window_extent(renderer).width / self.fig.dpi
            if cursor + width > x + w - 0.6 and j < len(seeds) - 1:
                chip.remove()
                L.text(cursor, y, f"+{len(seeds) - j} more", fontsize=7.5, color=MUTED, va="center")
                break
            chip.seed = seed
            chips.append(chip)
            cursor += width + 0.12
        return chips

    # --- choosing ----------------------------------------------------------------------------------

    def open(self, scenario: str, seed=None):
        """Open the comparison viewer on `scenario` (at `seed`, default its first) and close the
        picker. Returns the viewer."""
        from gridlock.viz.replay import ComparisonViewer

        picked = pick_comparison(self.trajectories, scenario, seed)
        self.viewer = ComparisonViewer(picked, interval_ms=self.interval_ms, on_back=self.reopen)
        self.viewer.fig.show(warn=False)              # show the new window before closing this one
        plt.close(self.fig)
        return self.viewer

    def reopen(self) -> None:
        """Bring the picker back in a new window (called by the viewer's "← levels")."""
        self.viewer = None
        self._open_window()
        self.fig.show(warn=False)

    def show(self) -> None:
        plt.show()

    def _hit(self, event):
        """(card index, chip or None) under the mouse, or (None, None)."""
        fx, fy = event.x / self.fig.bbox.width, event.y / self.fig.bbox.height
        for k, card in enumerate(self.cards):
            x0, y0, w, h = card["rect"]
            if x0 <= fx <= x0 + w and y0 <= fy <= y0 + h:
                chip = next((chip for chip in card["chips"] if chip.contains(event)[0]), None)
                return k, chip
        return None, None

    def _on_click(self, event) -> None:
        if event.button != 1:
            return
        k, chip = self._hit(event)
        if k is not None and self.cards[k]["recorded"]:
            self.open(self.cards[k]["name"], None if chip is None else chip.seed)

    def _on_move(self, event) -> None:
        k, chip = self._hit(event)
        if k is not None and not self.cards[k]["recorded"]:
            k = None
        if (k, chip) == (self.hover, self.hover_chip):
            return
        if self.hover is not None:
            self.cards[self.hover]["patch"].set_linewidth(theme.RULE_W)
        if self.hover_chip is not None:
            self.hover_chip.get_bbox_patch().set_facecolor(PAPER)
            self.hover_chip.set_color(INK)
        self.hover, self.hover_chip = k, chip
        if k is not None:
            self.cards[k]["patch"].set_linewidth(2.6)
        if chip is not None:
            chip.get_bbox_patch().set_facecolor(INK)
            chip.set_color(PAPER)
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key and event.key.isdigit() and event.key != "0":
            k = int(event.key) - 1
            if k < len(self.cards) and self.cards[k]["recorded"]:
                self.open(self.cards[k]["name"])

    def _on_resize(self, _event) -> None:
        size = tuple(self.fig.get_size_inches())
        if size != self._size:                        # text and glyphs don't scale with the window
            self._size = size
            self.fig.clear()
            self.hover, self.hover_chip = None, None
            self._build()
            self.fig.canvas.draw_idle()
