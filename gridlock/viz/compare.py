"""Several agents on the same level and seed, side by side and stepped together.

Each recorded episode has one agent under study (the scenario's first agent, by sorted id) and
possibly opponents. The comparison shows one card per episode: the studied agent in its own
colour, the path it has taken so far, where it broke R* (✕), and a status pill saying whether it
is still going, reached the goal or was cut off. Below are every agent's cumulative R and R* and
the level description.
"""

from itertools import accumulate

from matplotlib.lines import Line2D

from gridlock.env.grid import AGENT_COLORS, Grid
from gridlock.env.gridworld import object_positions
from gridlock.viz import theme
from gridlock.viz.panels import EVENT_ICONS, draw_text_lines, level_lines
from gridlock.viz.theme import ACCENT, FLAG, GOOD, INK, MUTED, WARN, Layout, themed

OPPONENT_COLORS = ("#64748b", "#94a3b8", "#475569", "#cbd5e1")
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
        L.text(MARGIN, 0.2, tr.scenario, fontsize=17, fontweight="bold", color=INK)
        L.text(MARGIN, 0.6, f"seed {tr.seed}  ·  {n} agent{'s' if n != 1 else ''}:  "
                            + ",  ".join(t.agent for t in self.trajs), fontsize=9, color=MUTED)
        self.frame_text = L.text(L.w - MARGIN, 0.42, "", fontsize=10, fontweight="bold", color=INK, ha="right",
                                 va="center")
        self.track, self.fill = theme.progress(L, MARGIN, 0.86, L.w - 2 * MARGIN, h=0.05)

        self.ax_boards, self.ax_logs, self.status_texts, self.score_texts = [], [], [], []
        for k, traj in enumerate(self.trajs):
            r, c = divmod(k, cols)
            x, y = MARGIN + c * (card_w + GAP), HEADER_H + r * (card_h + GAP)
            color = self.colors[k]
            L.card(x, y, card_w, card_h)
            L.bar(x + 0.12, y, card_w - 0.24, 0.05, color)                   # coloured top edge
            L.text(x + 0.16, y + 0.17, "●", fontsize=11, color=color)
            L.text(x + 0.4, y + 0.15, traj.agent, fontsize=11.5, fontweight="bold", color=INK)
            L.text(x + 0.4, y + 0.42, f"controls {self.studied[k]}", fontsize=7.5, color=MUTED)
            self.status_texts.append(L.text(x + card_w - 0.15, y + 0.27, "", fontsize=7.5, fontweight="bold",
                                            color="white", ha="right", va="center",
                                            bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.9", fc=MUTED, ec="none")))
            self.score_texts.append(L.text(x + card_w - 0.15, y + 0.5, "", fontsize=8.5, color=INK, ha="right"))
            log_h = 0.62
            ax = L.axes(x + 0.12, y + 0.78, card_w - 0.24, card_h - 0.78 - log_h - 0.12)
            self.grid.draw_board(ax)
            self.ax_boards.append(ax)
            self.ax_logs.append(L.axes(x + 0.2, y + card_h - log_h - 0.06, card_w - 0.4, log_h))

        y = HEADER_H + rows * (card_h + GAP)
        L.card(MARGIN, y, main_w, bottom - y)
        self.ax_returns = L.axes(MARGIN + 0.55, y + 0.55, main_w - 0.8, bottom - y - 0.95)

        side_x = L.w - MARGIN - SIDEBAR_W
        L.card(side_x, HEADER_H, SIDEBAR_W, bottom - HEADER_H)
        self.ax_level = L.axes(side_x + 0.22, HEADER_H + 0.42, SIDEBAR_W - 0.4, bottom - HEADER_H - 0.55)
        self._draw_level()

    def _draw_level(self) -> None:
        tr = self.trajs[0]
        draw_text_lines(self.fig, self.ax_level, level_lines(tr), wrap=True, mono=False, fontsize=8,
                        line_pts=11.5)
        theme.section_label(self.ax_level, f"Level  ·  {tr.scenario}")

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
            self.frame_text.set_text(f"frame {i} / {self.n_frames - 1}")
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
        """Status pill text and colour."""
        if status == "start":
            return "START", MUTED
        if status.startswith("reached"):
            return f"⚑ GOAL  t = {status.rsplit('= ', 1)[1]}", GOOD
        if status.startswith("cut off"):
            return f"CUT OFF  t = {status.rsplit('= ', 1)[1]}", WARN
        return f"RUNNING  {status}", ACCENT

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
        pieces += ax.plot(xs, ys, color=color, lw=4.5, alpha=0.35, solid_joinstyle="round", zorder=1.8)
        pieces += ax.plot(xs[:1], ys[:1], marker="o", ms=10, mfc="white", mec=color, mew=2.2, zorder=1.9)
        pieces += self.grid.draw_pieces(ax, agents, object_positions(state.get("objects", {})), colors)
        marked = set()
        for step in tr.steps[:self._frame(k, i)]:
            if step.get("violations"):
                x, y = step["state"]["agents"][aid]          # corner of the cell, so the agent stays visible
                if (x, y) in marked:
                    continue
                marked.add((x, y))
                pieces += ax.plot([x + 0.3], [y - 0.3], marker="o", ms=11, mfc="white", mec=WARN, mew=1.5, zorder=4)
                pieces.append(ax.text(x + 0.3, y - 0.29, "✕", ha="center", va="center", fontsize=7, color=WARN,
                                      fontweight="bold", zorder=4.1))
        flagged = any(flag["agent_id"] == aid and flag["t"] <= state.get("t", 0) for flag in tr.monitor_flags)
        if flagged and aid in agents:
            pieces += ax.plot(*agents[aid], marker="o", ms=26, mfc="none", mec=FLAG, mew=2.5, zorder=4.5)

        text, fc = self._pill(self._status(k, i))
        self.status_texts[k].set_text(text)
        self.status_texts[k].get_bbox_patch().set_facecolor(fc)
        r, r_star = self._totals(k, i)
        self.score_texts[k].set_text(f"R {r:+.1f}    R* {r_star:+.1f}")

    def _log_lines(self, k: int, i: int) -> list[tuple]:
        tr, f, aid = self.trajs[k], self._frame(k, i), self.studied[k]
        if f == 0:
            return [("○ start   ━ path so far   ✕ R* violation", MUTED)]
        step = tr.steps[f - 1]
        lines = []
        if f < i:
            lines.append(("episode over", MUTED))
        lines.append((f"t = {step['t']}  " + "  ".join(f"{a}: {act}" for a, act in sorted(step["actions"].items())),
                      INK))
        lines += [(f"✕ R* violated: {v}", WARN, True) for v in step.get("violations", [])]
        lines += [(f"⚑ FLAG {flag['agent_id']}: {flag['reason']}", FLAG, True) for flag in step.get("flags", [])]
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
            lw = 2 + 1.6 * (n - 1 - k)          # earlier agents thicker: identical runs show as nested bands
            ts = [tr.initial_state.get("t", 0)] + [s["t"] for s in tr.steps]
            r = list(accumulate((s["R"].get(aid, 0.0) for s in tr.steps), initial=0.0))
            r_star = list(accumulate((s.get("R_star", {}).get(aid, 0.0) for s in tr.steps), initial=0.0))
            ax.plot(ts, r, color=color, lw=lw, alpha=0.9)
            ax.plot(ts, r_star, color=color, lw=lw, ls=(0, (4, 3)), alpha=0.9)
            hits = [j for j, step in enumerate(tr.steps, start=1) if step.get("violations")]
            ms = 7 if len(hits) <= 10 else 3.5                 # many violations: keep the curve readable
            ax.plot([ts[j] for j in hits], [r_star[j] for j in hits], ls="", marker="X", ms=ms, color=color,
                    mec="white", mew=0.6, zorder=3)
            f = self._frame(k, i)
            ax.plot([ts[f]], [r[f]], "o", color=color, ms=6.5, mec="white", mew=1.5, zorder=5)
            ax.plot([ts[f]], [r_star[f]], "o", color="white", ms=6.5, mec=color, mew=1.8, zorder=5)
        ax.axhline(0, color=theme.CARD_EDGE, lw=1, zorder=0)
        ax.axvline(i, color=ACCENT, lw=1.5, alpha=0.8)
        handles = [Line2D([], [], color=self.colors[k], lw=3, label=tr.agent) for k, tr in enumerate(self.trajs)]
        handles += [Line2D([], [], color=MUTED, lw=2, label="R"),
                    Line2D([], [], color=MUTED, lw=2, ls=(0, (4, 3)), label="R*"),
                    Line2D([], [], color=MUTED, ls="", marker="X", mec="white", label="violation")]
        ax.legend(handles=handles, ncol=len(handles), loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=7.5,
                  handlelength=1.8, columnspacing=1.0, borderaxespad=0.2)
        ax.set_xlabel("t", fontsize=8, labelpad=1)
        theme.section_label(ax, "Cumulative R (solid) and R* (dashed)", pad_pts=8)
