"""Several agents on the same level and seed, side by side and stepped together.

Each recorded episode has one agent under study (the scenario's first agent, by sorted id) and
possibly opponents. The comparison shows one card per episode: the studied agent in its own
colour, the path it has taken so far, where it broke R* (✕), and a status box saying whether it
is still going, reached the goal or was cut off. Below are every agent's cumulative R and R* and
the level description.
"""

from itertools import accumulate

from matplotlib.lines import Line2D

from gridlock.env.grid import AGENT_COLORS, Grid
from gridlock.env.gridworld import object_positions
from gridlock.viz import theme
from gridlock.viz.panels import EVENT_ICONS, crosshair, draw_text_lines, level_lines
from gridlock.viz.theme import (ACCENT, GRAY, INK, INVERSE_INK, INVERSE_MUTED, MUTED, PAPER, WARN, Layout,
                                themed)

OPPONENT_COLORS = ("#6b6b6b", "#a5a5a5", "#3d3d3d", "#cfcfcf")
MAX_COLUMNS = 4
# Inches. The default window is 14 x 8, which fits a 13-inch laptop screen.
MARGIN, GAP, HEADER_H, CONTROLS_H, SIDEBAR_W, CARD_H = 0.25, 0.18, 1.02, 0.62, 3.7, 3.95


def figure_size(n: int) -> tuple[float, float]:
    cols = max(1, min(n, MAX_COLUMNS))
    rows = -(-n // cols)
    width = 14.0 if cols <= 3 else 15.5
    return width, 8.0 + (rows - 1) * (CARD_H + GAP)


class ComparisonDashboard:
    """One card per trajectory, all drawn at the same frame. Frame 0 is the state after reset and
    frame k the state after step k; an episode that has already ended stays on its last frame.

    Layout: header and progress bar on top, agent cards with a wide R/R* chart under them, and
    the level description in a sidebar on the right."""

    def __init__(self, trajs, fig):
        if not trajs:
            raise ValueError("nothing to compare")
        self.trajs = list(trajs)
        self.fig = fig
        self.grid = Grid.from_rows(self.trajs[0].grid_rows)
        self.colors = [AGENT_COLORS[k % len(AGENT_COLORS)] for k in range(len(self.trajs))]
        self.studied = [sorted(tr.initial_state.get("agents", {}))[0] for tr in self.trajs]
        self._pieces = [[] for _ in self.trajs]
        with themed():
            self._build()

    def _build(self) -> None:
        L = self.layout = Layout(self.fig)
        n = len(self.trajs)
        cols = min(n, MAX_COLUMNS)
        rows = -(-n // cols)
        bottom = L.h - CONTROLS_H - 0.12                  # cards end here; the control bar sits below
        main_w = L.w - 2 * MARGIN - SIDEBAR_W - GAP
        card_w = (main_w - (cols - 1) * GAP) / cols
        card_h = min(CARD_H, (bottom - HEADER_H - 1.6 - rows * GAP) / rows)

        tr = self.trajs[0]
        L.title(MARGIN - 0.03, 0.02, tr.scenario)
        L.text(MARGIN, 0.66, f"seed {tr.seed}  /  {n} agent{'s' if n != 1 else ''}:  "
                             + ",  ".join(t.agent for t in self.trajs), fontsize=8, color=INK)
        self.frame_text = L.title(L.w - MARGIN + 0.02, 0.1, "", size=22, ha="right")
        self.track, self.fill = theme.progress(L, MARGIN, 0.86, L.w - 2 * MARGIN, h=0.05)
        L.edge_label(f"gridlock  /  comparison  /  {tr.scenario}  /  seed {tr.seed}")

        self.ax_boards, self.ax_logs, self.status_texts, self.score_texts = [], [], [], []
        for k, traj in enumerate(self.trajs):
            r, c = divmod(k, cols)
            x, y = MARGIN + c * (card_w + GAP), HEADER_H + r * (card_h + GAP)
            color = self.colors[k]
            L.card(x, y, card_w, card_h)
            L.bar(x, y, card_w, 0.1, color)                                  # coloured top band
            L.text(x + 0.14, y + 0.2, f"{k + 1:02d}", fontsize=7.5, color=MUTED)
            L.title(x + 0.45, y + 0.12, traj.agent, size=17)
            L.text(x + 0.48, y + 0.5, f"controls {self.studied[k]}", fontsize=7, color=MUTED)
            self.status_texts.append(L.text(x + card_w - 0.16, y + 0.3, "", fontsize=7, color=PAPER, ha="right",
                                            va="center", bbox=theme.box(GRAY)))
            self.score_texts.append(L.text(x + card_w - 0.12, y + 0.52, "", fontsize=7.5, color=INK, ha="right"))
            L.rule(x, y + 0.74, card_w, lw=theme.HAIR_W)
            log_h = 0.62
            ax = L.axes(x + 0.14, y + 0.86, card_w - 0.28, card_h - 0.86 - log_h - 0.16)
            self.grid.draw_board(ax)
            self.ax_boards.append(ax)
            L.rule(x, y + card_h - log_h - 0.12, card_w, lw=theme.HAIR_W)
            self.ax_logs.append(L.axes(x + 0.14, y + card_h - log_h - 0.04, card_w - 0.28, log_h - 0.04))

        y = HEADER_H + rows * (card_h + GAP)
        L.card(MARGIN, y, main_w, bottom - y)
        self.ax_returns = L.axes(MARGIN + 0.55, y + 0.55, main_w - 0.8, bottom - y - 0.95)

        side_x = L.w - MARGIN - SIDEBAR_W
        L.card(side_x, HEADER_H, SIDEBAR_W, bottom - HEADER_H, inverse=True)
        self.ax_level = L.axes(side_x + 0.22, HEADER_H + 0.42, SIDEBAR_W - 0.4, bottom - HEADER_H - 0.55)
        self._draw_level()

    def _draw_level(self) -> None:
        tr = self.trajs[0]
        draw_text_lines(self.fig, self.ax_level, level_lines(tr), wrap=True, fontsize=7.1, line_pts=10,
                        rule_color=INVERSE_MUTED)
        theme.section_label(self.ax_level, f"level / {tr.scenario}", color=INVERSE_INK, index=len(self.trajs) + 2)

    def on_resize(self) -> None:
        """Re-fit what depends on the window size: the boards' symbols and the wrapped level text."""
        with themed():
            for k, ax in enumerate(self.ax_boards):
                ax.clear()
                self._pieces[k] = []
                self.grid.draw_board(ax)
            self._draw_level()

    @property
    def n_frames(self) -> int:
        return max(len(tr.steps) for tr in self.trajs) + 1

    def draw(self, i: int) -> int:
        i = max(0, min(i, self.n_frames - 1))
        with themed():
            self.frame_text.set_text(f"{i:02d}/{self.n_frames - 1:02d}")
            theme.set_progress(self.fill, self.track, i / max(1, self.n_frames - 1))
            for k in range(len(self.trajs)):
                self._draw_card(k, i)
                draw_text_lines(self.fig, self.ax_logs[k], self._log_lines(k, i), fontsize=7.5, line_pts=10.5)
            self._draw_returns(i)
        return i

    # --- per-episode helpers -------------------------------------------------------------------

    def _frame(self, k: int, i: int) -> int:
        return min(i, len(self.trajs[k].steps))

    def _state(self, k: int, i: int) -> dict:
        tr, f = self.trajs[k], self._frame(k, i)
        return tr.initial_state if f == 0 else tr.steps[f - 1]["state"]

    def _totals(self, k: int, i: int) -> tuple[float, float]:
        steps, aid = self.trajs[k].steps[:self._frame(k, i)], self.studied[k]
        return (sum(s["R"].get(aid, 0.0) for s in steps), sum(s.get("R_star", {}).get(aid, 0.0) for s in steps))

    def _status(self, k: int, i: int) -> str:
        tr, f = self.trajs[k], self._frame(k, i)
        if f == 0:
            return "start"
        t = tr.steps[f - 1]["t"]
        if f < len(tr.steps):
            return f"t = {t}"
        if self.studied[k] in tr.steps[-1]["state"].get("done", []):
            return f"reached the goal at t = {t}"
        return f"cut off at t = {t}"

    @staticmethod
    def _pill(status: str) -> tuple[str, str]:
        """Status box text and fill: gray before the start, black once at the goal, white with a
        black rule while running or cut off (the text colour follows the fill)."""
        if status == "start":
            return "START", GRAY
        if status.startswith("reached"):
            return f"⚑ GOAL  t = {status.rsplit('= ', 1)[1]}", INK
        if status.startswith("cut off"):
            return f"CUT OFF  t = {status.rsplit('= ', 1)[1]}", PAPER
        return f"RUNNING  {status}", PAPER

    def _draw_card(self, k: int, i: int) -> None:
        ax, tr, aid, color = self.ax_boards[k], self.trajs[k], self.studied[k], self.colors[k]
        pieces = self._pieces[k]
        theme.clear_artists(pieces)
        state = self._state(k, i)
        agents = {a: tuple(pos) for a, pos in state.get("agents", {}).items()}
        others = sorted(a for a in tr.initial_state.get("agents", {}) if a != aid)
        colors = {aid: color, **{a: OPPONENT_COLORS[j % len(OPPONENT_COLORS)] for j, a in enumerate(others)}}

        path = [tuple(self._state(k, f).get("agents", {}).get(aid, (0, 0))) for f in range(self._frame(k, i) + 1)]
        xs, ys = [p[0] for p in path], [p[1] for p in path]
        pieces += ax.plot(xs, ys, color=color, lw=1.6, marker="s", ms=3, zorder=1.8)
        pieces += ax.plot(xs[:1], ys[:1], marker="s", ms=9, mfc="white", mec=color, mew=1.6, zorder=1.9)
        pieces += self.grid.draw_pieces(ax, agents, object_positions(state.get("objects", {})), colors)
        marked = set()
        for step in tr.steps[:self._frame(k, i)]:
            if step.get("violations"):
                x, y = step["state"]["agents"][aid]          # corner of the cell, so the agent stays visible
                if (x, y) in marked:
                    continue
                marked.add((x, y))
                pieces += ax.plot([x + 0.3], [y - 0.3], marker="s", ms=10, mfc="white", mec=WARN, mew=1.2, zorder=4)
                pieces.append(ax.text(x + 0.3, y - 0.29, "✕", ha="center", va="center", fontsize=7, color=WARN,
                                      zorder=4.1))
        flagged = any(flag["agent_id"] == aid and flag["t"] <= state.get("t", 0) for flag in tr.monitor_flags)
        if flagged and aid in agents:
            pieces += crosshair(ax, agents[aid])

        text, fc = self._pill(self._status(k, i))
        self.status_texts[k].set_text(text)
        self.status_texts[k].get_bbox_patch().set_facecolor(fc)
        self.status_texts[k].set_color(PAPER if fc == INK else INK)
        r, r_star = self._totals(k, i)
        self.score_texts[k].set_text(f"r {r:+.1f}    r* {r_star:+.1f}")

    def _log_lines(self, k: int, i: int) -> list[tuple]:
        tr, f, aid = self.trajs[k], self._frame(k, i), self.studied[k]
        if f == 0:
            return [("□ start   ━ path so far   ✕ r* violation", MUTED)]
        step = tr.steps[f - 1]
        lines = []
        if f < i:
            lines.append(("episode over", MUTED))
        lines.append((f"t = {step['t']}  " + "  ".join(f"{a}: {act.lower()}" for a, act in sorted(step["actions"].items())),
                      INK))
        lines += [(f"✕ r* violated: {v}", WARN, True) for v in step.get("violations", [])]
        lines += [(f"⚑ flag {flag['agent_id']}: {flag['reason']}", theme.FLAG, True) for flag in step.get("flags", [])]
        for event in step["events"]:
            if event["kind"] == "MOVED":
                continue
            target = f" → {event['target']}" if event.get("target") else ""
            lines.append((f"{EVENT_ICONS.get(event['kind'], '•')} {event['kind'].lower()} {event['agent_id']}{target}",
                          self.colors[k] if event["agent_id"] == aid else MUTED))
        return lines

    def _draw_returns(self, i: int) -> None:
        ax = self.ax_returns
        ax.clear()
        theme.style_chart(ax)
        n = len(self.trajs)
        for k, tr in enumerate(self.trajs):
            aid, color = self.studied[k], self.colors[k]
            lw = 1.4 + 1.8 * (n - 1 - k)        # earlier agents thicker: identical runs show as nested bands
            ts = [tr.initial_state.get("t", 0)] + [s["t"] for s in tr.steps]
            r = list(accumulate((s["R"].get(aid, 0.0) for s in tr.steps), initial=0.0))
            r_star = list(accumulate((s.get("R_star", {}).get(aid, 0.0) for s in tr.steps), initial=0.0))
            ax.plot(ts, r, color=color, lw=lw)
            ax.plot(ts, r_star, color=color, lw=lw, ls=(0, (4, 3)))
            hits = [j for j, step in enumerate(tr.steps, start=1) if step.get("violations")]
            ms = 7 if len(hits) <= 10 else 3.5                 # many violations: keep the curve readable
            ax.plot([ts[j] for j in hits], [r_star[j] for j in hits], ls="", marker="x", ms=ms, color=INK,
                    mew=1.2, zorder=3)
            f = self._frame(k, i)
            ax.plot([ts[f]], [r[f]], "s", color=color, ms=6, mec=INK, mew=1, zorder=5)
            ax.plot([ts[f]], [r_star[f]], "s", color="white", ms=6, mec=color, mew=1.4, zorder=5)
        ax.axhline(0, color=INK, lw=theme.HAIR_W, zorder=0)
        ax.axvline(i, color=ACCENT, lw=1)
        handles = [Line2D([], [], color=self.colors[k], lw=3, label=tr.agent) for k, tr in enumerate(self.trajs)]
        handles += [Line2D([], [], color=INK, lw=1.4, label="r"),
                    Line2D([], [], color=INK, lw=1.4, ls=(0, (4, 3)), label="r*"),
                    Line2D([], [], color=INK, ls="", marker="x", mew=1.2, label="violation")]
        ax.legend(handles=handles, ncol=len(handles), loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=7.5,
                  handlelength=1.8, columnspacing=1.0, borderaxespad=0.2)
        ax.set_xlabel("t", fontsize=8, labelpad=1)
        theme.section_label(ax, "cumulative r (solid) and r* (dashed)", pad_pts=8, index=len(self.trajs) + 1)
