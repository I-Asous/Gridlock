"""Matplotlib panels shared by the replay and live viewers.

They read only the recorded Trajectory (never IntentFn or the monitor), so any saved run can be
viewed later and nothing here can leak R* back into an agent.
"""

import textwrap
from itertools import accumulate

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy

from gridlock.env.grid import Grid, agent_palette
from gridlock.env.gridworld import object_positions
from gridlock.eval.metrics import time_to_detect
from gridlock.viz import theme
from gridlock.viz.theme import (ACCENT, FLAG, GRAY, INK, INVERSE_INK, INVERSE_MUTED, LOOPHOLE, MUTED, PAPER, WARN,
                                YELLOW, Layout, themed)

# Kept for older imports.
FLAG_COLOR, LOOPHOLE_COLOR, TEXT_COLOR, MUTED_COLOR = FLAG, LOOPHOLE, INK, MUTED
LOG_LINE_PTS = 12
LOG_FONT_PTS = 8

EVENT_ICONS = {"MOVED": "→", "PUSHED": "⇢", "BROKE": "✕", "REACHED_GOAL": "⚑", "CHECKPOINT": "★", "TRAPPED": "⚠",
               "BUMPED": "·", "BLOCKED": "⊘", "TOGGLED": "◐", "PICKED_UP": "↑"}


class Dashboard:
    """Board, level description, returns, flag timeline and event log for one trajectory, drawn
    at a chosen frame.

    Frame 0 is the state right after reset; frame k is the state after step k.
    """

    def __init__(self, traj, fig):
        self.traj = traj
        self.fig = fig
        self.grid = Grid.from_rows(traj.grid_rows)
        self.agent_ids = sorted(traj.initial_state.get("agents", {}))
        self.colors = agent_palette(self.agent_ids)
        self._pieces: list = []
        with themed():
            self._build()

    def _build(self) -> None:
        """Three columns under a header: the board with its step log, the charts, and the level
        description in a black sidebar. Sized for a 14 x 8 inch window; it scales with the window."""
        L = self.layout = Layout(self.fig)
        W, H, m, gap, side_w = L.w, L.h, 0.25, 0.18, 3.45
        top, bottom = 1.02, H - 0.62 - 0.12               # panels between the header and the control bar
        board_w = min(4.85, (bottom - top) - 1.15 + 0.3)  # the board panel is about as wide as it is tall
        mid_x = m + board_w + gap
        mid_w = W - 2 * m - board_w - side_w - 2 * gap
        side_x = W - m - side_w

        tr = self.traj
        L.title(m - 0.03, 0.02, tr.scenario)
        L.text(m, 0.66, f"{tr.agent}  /  seed {tr.seed}  /  {len(tr.steps)} steps", fontsize=8, color=INK)
        self.chips = [L.text(W - m, 0.5, "", fontsize=8, ha="right", va="center", color=fg, bbox=theme.box(fc))
                      for fc, fg in ((PAPER, INK), (PAPER, INK), (FLAG, PAPER), (GRAY, INK))]
        self.track, self.fill = theme.progress(L, m, 0.86, W - 2 * m, h=0.05)
        L.edge_label(f"gridlock  /  replay  /  {tr.scenario}  /  seed {tr.seed}")

        log_h = 1.05
        L.card(m, top, board_w, bottom - top)
        self.board_title = L.text(m + 0.16, top + 0.14, "", fontsize=8, color=INK)
        L.rule(m, top + 0.38, board_w, lw=theme.HAIR_W)
        self.ax_grid = L.axes(m + 0.16, top + 0.5, board_w - 0.32, bottom - top - 0.5 - log_h - 0.1)
        self.grid.draw_board(self.ax_grid)
        L.rule(m, bottom - log_h - 0.12, board_w, lw=theme.HAIR_W)
        self.ax_log = L.axes(m + 0.16, bottom - log_h - 0.02, board_w - 0.32, log_h - 0.12)

        returns_h = (bottom - top - gap) * 0.68
        L.card(mid_x, top, mid_w, returns_h)
        self.ax_returns = L.axes(mid_x + 0.5, top + 0.62, mid_w - 0.7, returns_h - 0.85)
        y = top + returns_h + gap
        L.card(mid_x, y, mid_w, bottom - y)
        self.ax_timeline = L.axes(mid_x + 0.5, y + 0.42, mid_w - 0.7, bottom - y - 0.8, sharex=self.ax_returns)

        L.card(side_x, top, side_w, bottom - top, inverse=True)
        self.ax_level = L.axes(side_x + 0.22, top + 0.42, side_w - 0.4, bottom - top - 0.55)
        self._draw_level()

    def _draw_level(self) -> None:
        draw_text_lines(self.fig, self.ax_level, self._level_lines(), wrap=True, fontsize=6.9, line_pts=9.6,
                        rule_color=INVERSE_MUTED)
        theme.section_label(self.ax_level, f"level / {self.traj.scenario}", color=INVERSE_INK, index=4)

    def on_resize(self) -> None:
        """Re-fit what depends on the window size: the board's symbols and the wrapped level text."""
        with themed():
            self.ax_grid.clear()
            self._pieces = []
            self.grid.draw_board(self.ax_grid)
            self._draw_level()

    @property
    def n_frames(self) -> int:
        return len(self.traj.steps) + 1

    def state_at(self, i: int) -> dict:
        return self.traj.initial_state if i == 0 else self.traj.steps[i - 1]["state"]

    def t_at(self, i: int) -> int:
        return self.state_at(i).get("t", i)

    def draw(self, i: int) -> int:
        """Redraw every panel at frame i (clamped to the episode) and return the frame drawn."""
        i = max(0, min(i, self.n_frames - 1))
        with themed():
            self._draw_header(i)
            self._draw_grid(i)
            self._draw_returns(i)
            self._draw_timeline(i)
            draw_text_lines(self.fig, self.ax_log, self._log_lines(i), fontsize=7.5, line_pts=10.5)
            first, last = self.t_at(0), self.t_at(self.n_frames - 1)
            self.ax_returns.set_xlim(first - 0.5, max(last, first + 1) + 0.5)
        return i

    def _draw_header(self, i: int) -> None:
        tr, steps = self.traj, self.traj.steps[:i]
        r = sum(sum(s["R"].values()) for s in steps)
        r_star = sum(sum(s.get("R_star", {}).values()) for s in steps)
        flags = sum(len(s.get("flags", [])) for s in steps)
        loopholes = sum(1 for t in tr.ground_truth_loophole_steps if t <= self.t_at(i))
        for chip, text in zip(self.chips, (f"r {r:+.1f}", f"r* {r_star:+.1f}", f"flags {flags}",
                                           f"loophole steps {loopholes}")):
            chip.set_text(text)
        flag_chip = self.chips[2]
        flag_chip.get_bbox_patch().set_facecolor(FLAG if flags else PAPER)
        flag_chip.set_color(PAPER if flags else INK)
        self._space_chips()
        theme.set_progress(self.fill, self.track, i / max(1, self.n_frames - 1))

    def _space_chips(self) -> None:
        """Lay the chips out right to left using their rendered widths."""
        renderer = self.fig.canvas.get_renderer()
        x = 1 - 0.25 / self.layout.w
        for chip in reversed(self.chips):           # chips read R, R*, flags, loopholes
            chip.set_x(x)
            width = chip.get_window_extent(renderer).width / self.fig.bbox.width
            x -= width + 0.006

    def _draw_grid(self, i: int) -> None:
        ax = self.ax_grid
        theme.clear_artists(self._pieces)
        state = self.state_at(i)
        agents = {aid: tuple(pos) for aid, pos in state.get("agents", {}).items()}
        for aid in self.agent_ids:
            path = [tuple(self.state_at(f).get("agents", {}).get(aid, (0, 0))) for f in range(i + 1)]
            self._pieces += ax.plot([p[0] for p in path], [p[1] for p in path], color=self.colors[aid], lw=1.6,
                                    marker="s", ms=3, zorder=1.8)
            self._pieces += ax.plot(*path[0], marker="s", ms=10, mfc="white", mec=self.colors[aid], mew=1.6,
                                    zorder=1.9)                     # where it started
        self._pieces += self.grid.draw_pieces(ax, agents, object_positions(state.get("objects", {})), self.colors)
        t = self.t_at(i)
        flagged = {flag["agent_id"] for flag in self.traj.monitor_flags if flag["t"] <= t}
        for aid in sorted(flagged & agents.keys()):
            self._pieces += crosshair(ax, agents[aid])
        title = f"01 / board    t = {t}    frame {i} / {self.n_frames - 1}"
        if state.get("flags"):
            title += "    /    " + ", ".join(state["flags"])
        self.board_title.set_text(title)

    def _mark_time(self, ax, i: int) -> None:
        for t in self.traj.ground_truth_loophole_steps:
            ax.axvspan(t - 0.5, t + 0.5, fc="none", ec=LOOPHOLE, hatch="////", lw=0, zorder=0)
        ax.axvline(self.t_at(i), color=ACCENT, lw=1)

    def _draw_returns(self, i: int) -> None:
        ax = self.ax_returns
        ax.clear()
        theme.style_chart(ax)
        steps = self.traj.steps
        ts = [self.t_at(k) for k in range(self.n_frames)]
        for aid in self.agent_ids:
            color = self.colors[aid]
            r = list(accumulate((s["R"].get(aid, 0.0) for s in steps), initial=0.0))
            r_star = list(accumulate((s.get("R_star", {}).get(aid, 0.0) for s in steps), initial=0.0))
            ax.plot(ts, r, color=color, lw=1.6, label=f"{aid} r")
            ax.plot(ts, r_star, color=color, lw=1.6, ls=(0, (4, 3)), label=f"{aid} r*")
            ax.plot([ts[i]], [r[i]], "s", color=color, ms=6, mec=INK, mew=1, zorder=4)
            ax.plot([ts[i]], [r_star[i]], "s", color="white", ms=6, mec=color, mew=1.4, zorder=4)
        ax.axhline(0, color=INK, lw=theme.HAIR_W, zorder=0)
        self._mark_time(ax, i)
        ax.tick_params(labelbottom=False)
        if self.agent_ids:
            ax.legend(ncol=min(4, 2 * len(self.agent_ids)), loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=7.5,
                      handlelength=2.2, borderaxespad=0.2)
        theme.section_label(ax, "cumulative r (solid) and r* (dashed)", pad_pts=8, index=2)

    def _draw_timeline(self, i: int) -> None:
        ax = self.ax_timeline
        ax.clear()
        theme.style_chart(ax)
        ax.grid(False)
        flags = self.traj.monitor_flags
        rows = self.agent_ids + sorted({flag["agent_id"] for flag in flags} - set(self.agent_ids))
        for row, aid in enumerate(rows):
            ax.axhline(row, color=INK, lw=theme.HAIR_W, ls=(0, (1, 2)), zorder=0)
            ts = [flag["t"] for flag in flags if flag["agent_id"] == aid]
            ax.scatter(ts, [row] * len(ts), marker="v", s=52, color=FLAG, edgecolors="none", zorder=3)
        self._mark_time(ax, i)
        ax.set_yticks(range(len(rows)), rows)
        ax.set_ylim(max(len(rows), 1) - 0.5, -0.5)
        ax.set_xlabel("t", fontsize=8)

        loopholes = self.traj.ground_truth_loophole_steps
        ttd = time_to_detect(min(loopholes, default=None), min((f["t"] for f in flags), default=None))
        ttd_text = "n/a" if ttd is None else f"{ttd:+d}"
        theme.section_label(ax, f"monitor flags ▼  /  time to detect {ttd_text}", pad_pts=8, index=3)

    def _level_lines(self) -> list[tuple]:
        return level_lines(self.traj)

    def _log_lines(self, i: int) -> list[tuple]:
        if i == 0:
            return [("initial state, after reset", MUTED)]
        step = self.traj.steps[i - 1]
        lines = [(f"t = {step['t']}", INK, True)]
        lines += [(f"⚑ flag  {f['agent_id']}: {f['reason']}", FLAG, True) for f in step.get("flags", [])]
        lines += [(f"✕ r* violated: {v}", WARN, True) for v in step.get("violations", [])]
        for aid in self.agent_ids:
            r, r_star = step["R"].get(aid, 0.0), step.get("R_star", {}).get(aid, 0.0)
            action = step["actions"].get(aid, "")
            lines.append((f"{aid:<8}{action.lower():<9} r {r:+6.2f}   r* {r_star:+6.2f}", self.colors[aid]))
        for event in step["events"]:
            target = f" → {event['target']}" if event.get("target") else ""
            icon = EVENT_ICONS.get(event["kind"], "•")
            lines.append((f"  {icon} {event['kind'].lower():<13}{event['agent_id']} @ {tuple(event['pos'])}{target}",
                          self.colors.get(event["agent_id"], MUTED)))
        return lines


def draw_text_lines(fig, ax, lines: list[tuple], wrap: bool = False, mono: bool = True,
                    fontsize: float = LOG_FONT_PTS, line_pts: float = LOG_LINE_PTS, rule_color=None) -> None:
    """Write (text, color[, heading]) lines top-down on `ax`, wrapping them to its width when
    `wrap` is set and cutting them off with "… N more" when they don't fit.

    Nothing is bold. With `rule_color`, each heading line after the first gets a hairline rule
    above it instead, like the breakers between sections on a ruled sheet."""
    ax.clear()
    ax.axis("off")
    family = theme.mono() if mono else theme.sans()
    box = ax.get_window_extent()
    if wrap:
        char_pts = fontsize * (0.6 if mono else 0.47)
        width = max(20, int(box.width * 72 / fig.dpi // char_pts))
        wrapped = []
        for text, color, *heading in lines:
            indent = " " * (len(text) - len(text.lstrip(" •")) + (1 if text.lstrip().startswith("•") else 0))
            for k, piece in enumerate(textwrap.wrap(text, width, subsequent_indent=indent) or [""]):
                wrapped.append((piece, color, *(heading if k == 0 else [])))
        lines = wrapped
    gap = 0.7 if rule_color is not None else 0.0     # extra lines of space above a ruled heading
    tops, y = [], 0.0
    for k, (_text, _color, *heading) in enumerate(lines):
        if k and heading and heading[0]:
            y += gap
        tops.append(y)
        y += 1
    room = box.height * 72 / fig.dpi / line_pts      # in lines
    if tops and tops[-1] + 1 > room:
        keep = max(1, sum(1 for top in tops if top + 2 <= room))
        hidden = len(lines) - keep
        lines = lines[:keep] + [(f"… {hidden} more", MUTED if rule_color is None else rule_color)]
        tops = tops[:keep] + [tops[keep - 1] + 1]
    for k, ((text, color, *heading), top) in enumerate(zip(lines, tops)):
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, -top * line_pts),
                    textcoords="offset points", va="top", ha="left", family=family, fontsize=fontsize, color=color)
        if gap and k and heading and heading[0]:
            at = offset_copy(ax.transAxes, fig=fig, y=-(top - gap / 2 - 0.1) * line_pts, units="points")
            ax.add_line(Line2D([0, 1], [1, 1], transform=at, color=rule_color, lw=theme.HAIR_W, clip_on=False))


def crosshair(ax, pos) -> list:
    """Red hairlines through a cell across the whole board and a red square around it: how the
    viewers mark an agent the monitor has flagged."""
    x, y = pos
    return [ax.axhline(y, color=FLAG, lw=0.8, zorder=2.8), ax.axvline(x, color=FLAG, lw=0.8, zorder=2.8),
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec=FLAG, lw=2.2, zorder=4))]


def level_lines(traj) -> list[tuple]:
    """The recorded level description as (text, color[, heading]) lines, coloured for the black
    sidebar: white text, gray bullets, numbered lowercase headings, loopholes in yellow."""
    level = traj.level
    if not level:
        return [(f"{traj.scenario}", INVERSE_INK, True),
                ("No level description recorded. Re-run the experiment to record one.", INVERSE_MUTED)]
    lines = []
    if level.get("description"):
        lines.append((level["description"], INVERSE_INK))
    sections = (("rules", "rules", INVERSE_MUTED, INVERSE_INK),
                ("r  —  stated reward, what agents optimise", "reward", INVERSE_MUTED, INVERSE_INK),
                ("r*  —  hidden intent, agents never see it", "intent", INVERSE_MUTED, INVERSE_INK),
                ("designed loopholes", "loopholes", YELLOW, YELLOW))
    for n, (title, key, color, heading) in enumerate(sections, start=1):
        if level.get(key):
            lines.append((f"{chr(96 + n)}  {title}", heading, True))
            lines += [(f" • {item}", color) for item in level[key]]
    return lines


__all__ = ["Dashboard", "crosshair", "draw_text_lines", "level_lines", "Line2D"]
