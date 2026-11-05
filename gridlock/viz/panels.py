"""Matplotlib panels shared by the replay and live viewers.

They read only the recorded Trajectory (never IntentFn or the monitor), so any saved run can be
viewed later and nothing here can leak R* back into an agent.
"""

from itertools import accumulate

from matplotlib.patches import Circle

from gridlock.env.grid import Grid, agent_palette
from gridlock.env.gridworld import object_positions
from gridlock.eval.metrics import time_to_detect

FLAG_COLOR = "#d62728"
LOOPHOLE_COLOR = "#9ca3af"
TEXT_COLOR = "#111827"
MUTED_COLOR = "#6b7280"
LOG_LINE_PTS = 11


class Dashboard:
    """Grid, returns, flag timeline and event log for one trajectory, drawn at a chosen frame.

    Frame 0 is the state right after reset; frame k is the state after step k.
    """

    def __init__(self, traj, fig):
        self.traj = traj
        self.fig = fig
        self.grid = Grid.from_rows(traj.grid_rows)
        self.agent_ids = sorted(traj.initial_state.get("agents", {}))
        self.colors = agent_palette(self.agent_ids)

        gs = fig.add_gridspec(3, 2, width_ratios=(1.1, 1), height_ratios=(1.1, 0.6, 1.1),
                              left=0.03, right=0.98, top=0.88, bottom=0.17, wspace=0.22, hspace=0.55)
        self.ax_grid = fig.add_subplot(gs[:, 0])
        self.ax_returns = fig.add_subplot(gs[0, 1])
        self.ax_timeline = fig.add_subplot(gs[1, 1], sharex=self.ax_returns)
        self.ax_log = fig.add_subplot(gs[2, 1])

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
        self._draw_summary()
        self._draw_grid(i)
        self._draw_returns(i)
        self._draw_timeline(i)
        self._draw_log(i)
        first, last = self.t_at(0), self.t_at(self.n_frames - 1)
        self.ax_returns.set_xlim(first - 0.5, max(last, first + 1) + 0.5)
        return i

    def _draw_summary(self) -> None:
        tr = self.traj
        self.fig.suptitle(
            f"R = {tr.total_R:+.2f}     R* = {tr.total_R_star:+.2f}     "
            f"monitor flags: {len(tr.monitor_flags)}     loophole steps: {len(tr.ground_truth_loophole_steps)}",
            fontsize=10)

    def _draw_grid(self, i: int) -> None:
        ax = self.ax_grid
        ax.clear()
        state = self.state_at(i)
        agents = {aid: tuple(pos) for aid, pos in state.get("agents", {}).items()}
        self.grid.render_matplotlib(agents, ax=ax, objects=object_positions(state.get("objects", {})),
                                    colors=self.colors)
        t = self.t_at(i)
        flagged = {flag["agent_id"] for flag in self.traj.monitor_flags if flag["t"] <= t}
        for aid in sorted(flagged & agents.keys()):
            ax.add_patch(Circle(agents[aid], 0.47, fill=False, ec=FLAG_COLOR, lw=2.5, zorder=4))
        title = f"t = {t}    (frame {i} / {self.n_frames - 1})"
        if state.get("flags"):
            title += "\nstate: " + ", ".join(state["flags"])
        ax.set_title(title, fontsize=10)

    def _mark_time(self, ax, i: int) -> None:
        for t in self.traj.ground_truth_loophole_steps:
            ax.axvspan(t - 0.5, t + 0.5, color=LOOPHOLE_COLOR, alpha=0.3, lw=0)
        ax.axvline(self.t_at(i), color=TEXT_COLOR, lw=1, alpha=0.6)

    def _draw_returns(self, i: int) -> None:
        ax = self.ax_returns
        ax.clear()
        steps = self.traj.steps
        ts = [self.t_at(k) for k in range(self.n_frames)]
        for aid in self.agent_ids:
            color = self.colors[aid]
            r = accumulate((s["R"].get(aid, 0.0) for s in steps), initial=0.0)
            r_star = accumulate((s.get("R_star", {}).get(aid, 0.0) for s in steps), initial=0.0)
            ax.plot(ts, list(r), color=color, lw=1.6, label=f"{aid} R")
            ax.plot(ts, list(r_star), color=color, lw=1.6, ls="--", label=f"{aid} R*")
        self._mark_time(ax, i)
        ax.set_title("cumulative R (solid) vs R* (dashed)", fontsize=9)
        ax.tick_params(labelbottom=False, labelsize=8)
        if self.agent_ids:
            ax.legend(fontsize=7, ncol=2, loc="best")

    def _draw_timeline(self, i: int) -> None:
        ax = self.ax_timeline
        ax.clear()
        flags = self.traj.monitor_flags
        rows = self.agent_ids + sorted({flag["agent_id"] for flag in flags} - set(self.agent_ids))
        for row, aid in enumerate(rows):
            ts = [flag["t"] for flag in flags if flag["agent_id"] == aid]
            ax.scatter(ts, [row] * len(ts), marker="v", s=40, color=FLAG_COLOR, zorder=3)
        self._mark_time(ax, i)
        ax.set_yticks(range(len(rows)), rows, fontsize=8)
        ax.set_ylim(max(len(rows), 1) - 0.5, -0.5)
        ax.set_xlabel("t", fontsize=8)
        ax.tick_params(labelsize=8)

        loopholes = self.traj.ground_truth_loophole_steps
        ttd = time_to_detect(min(loopholes, default=None), min((f["t"] for f in flags), default=None))
        ttd_text = "n/a" if ttd is None else f"{ttd:+d}"
        ax.set_title(f"monitor flags (▼) vs R* loophole steps (grey)    time-to-detect: {ttd_text}", fontsize=9)

    def _draw_log(self, i: int) -> None:
        ax = self.ax_log
        ax.clear()
        ax.axis("off")
        lines = self._log_lines(i)
        height_pts = ax.get_window_extent().height * 72 / self.fig.dpi
        max_lines = max(2, int(height_pts // LOG_LINE_PTS))
        if len(lines) > max_lines:
            hidden = len(lines) - max_lines + 1
            lines = lines[:max_lines - 1] + [(f"… {hidden} more", MUTED_COLOR)]
        for k, (text, color) in enumerate(lines):
            ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, -k * LOG_LINE_PTS),
                        textcoords="offset points", va="top", ha="left", family="monospace",
                        fontsize=8, color=color)

    def _log_lines(self, i: int) -> list[tuple[str, str]]:
        if i == 0:
            return [("initial state (after reset)", TEXT_COLOR)]
        step = self.traj.steps[i - 1]
        lines = [(f"step t={step['t']}", TEXT_COLOR)]
        lines += [(f"FLAG  {f['agent_id']}: {f['reason']}", FLAG_COLOR) for f in step.get("flags", [])]
        lines += [(f"R* violated: {v}", MUTED_COLOR) for v in step.get("violations", [])]
        actions = "  ".join(f"{aid}:{action}" for aid, action in sorted(step["actions"].items()))
        lines.append((f"actions  {actions}", TEXT_COLOR))
        for aid in self.agent_ids:
            r, r_star = step["R"].get(aid, 0.0), step.get("R_star", {}).get(aid, 0.0)
            lines.append((f"  {aid:<10} R {r:+6.2f}   R* {r_star:+6.2f}", self.colors[aid]))
        for event in step["events"]:
            target = f" → {event['target']}" if event.get("target") else ""
            lines.append((f"  {event['kind']:<12} {event['agent_id']} @ {tuple(event['pos'])}{target}",
                          TEXT_COLOR))
        return lines
