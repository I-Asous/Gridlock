"""Step through a recorded episode: slider, play/pause, arrow keys, and GIF/PNG export."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, Slider

from gridlock.viz import theme
from gridlock.viz.compare import ComparisonDashboard, figure_size
from gridlock.viz.panels import Dashboard
from gridlock.viz.theme import ACCENT, ACCENT_DARK, GRIDLINE, INK, MUTED, Layout, themed


class ReplayViewer:
    """Interactive viewer for one Trajectory. Keys: ←/→ step, space play/pause, home/end jump."""

    def __init__(self, traj, interval_ms: int = 300):
        self._setup(lambda fig: Dashboard(traj, fig), (14, 8), interval_ms)

    def _setup(self, make_dashboard, figsize, interval_ms: int) -> None:
        with themed():
            self.fig = plt.figure(figsize=figsize)
            if self.fig.canvas.manager is not None:
                self.fig.canvas.manager.set_window_title("Gridlock replay")
            self.dash = make_dashboard(self.fig)
            self.last = self.dash.n_frames - 1
            self.frame = 0
            self.playing = False
            self._build_controls()

        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._size = tuple(self.fig.get_size_inches())
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)
        self.dash.draw(0)

    def _build_controls(self) -> None:
        """A bar along the bottom: play button, frame slider and key hints. Hidden when saving."""
        L = Layout(self.fig)
        m, y, h = 0.25, L.h - 0.6, 0.48
        card = L.card(m, y, L.w - 2 * m, h)
        button_ax = L.axes(m + 0.12, y + 0.08, 1.0, h - 0.16)
        slider_ax = L.axes(m + 1.4, y + 0.14, L.w - 2 * m - 5.4, h - 0.28)
        self.button = Button(button_ax, "▶  play", color=ACCENT, hovercolor=ACCENT_DARK)
        self.button.label.set_color("white")
        for spine in button_ax.spines.values():
            spine.set_visible(False)
        self.button.on_clicked(lambda _event: self.toggle())
        self.slider = Slider(slider_ax, "", 0, max(self.last, 1), valinit=0, valstep=1, color=ACCENT,
                             track_color=GRIDLINE, initcolor="none",
                             handle_style={"facecolor": "white", "edgecolor": INK, "size": 12})
        handle = getattr(self.slider, "_handle", None)
        if handle is not None:
            handle.set_marker("s")                   # square handle; matplotlib only offers a circle option
            handle.set_markeredgewidth(1.2)
        self.slider.valtext.set_color(INK)
        self.slider.on_changed(self._on_slider)
        hint = L.text(L.w - m - 0.2, y + h / 2, "← →  step   /   space  play·pause   /   home end  jump",
                      fontsize=7.5, color=MUTED, ha="right", va="center")
        self._controls = (*card, slider_ax, button_ax, hint)

    def go(self, i: int) -> None:
        self.slider.set_val(max(0, min(i, self.last)))   # redraws through _on_slider

    def toggle(self) -> None:
        self.playing = not self.playing
        if self.playing:
            if self.frame >= self.last:
                self.go(0)
            self.timer.start()
        else:
            self.timer.stop()
        self.button.label.set_text("❚❚  pause" if self.playing else "▶  play")
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()

    def save(self, path, frame: Optional[int] = None, fps: int = 4) -> None:
        """A .gif path saves every frame as an animation; any other image path saves one
        frame (default: the last)."""
        path = Path(path)
        for artist in self._controls:
            artist.set_visible(False)
        try:
            if path.suffix.lower() == ".gif":
                anim = FuncAnimation(self.fig, self.dash.draw, frames=self.dash.n_frames, repeat=False)
                anim.save(path, writer=PillowWriter(fps=fps))
            else:
                self.dash.draw(self.last if frame is None else frame)
                self.fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.1)   # no empty control bar
        finally:
            for artist in self._controls:
                artist.set_visible(True)
            self.dash.draw(self.frame)

    def _on_resize(self, _event) -> None:
        size = tuple(self.fig.get_size_inches())
        if size != self._size:                      # text and symbols don't scale with the window
            self._size = size
            self.dash.on_resize()
            self.dash.draw(self.frame)
            self.fig.canvas.draw_idle()

    def _on_slider(self, value) -> None:
        self.frame = self.dash.draw(int(value))
        self.fig.canvas.draw_idle()

    def _tick(self) -> None:
        if self.frame >= self.last:
            self.toggle()
        else:
            self.go(self.frame + 1)

    def _on_key(self, event) -> None:
        jumps = {"right": self.frame + 1, "left": self.frame - 1, "home": 0, "end": self.last}
        if event.key == " ":
            self.toggle()
        elif event.key in jumps:
            self.go(jumps[event.key])


class ComparisonViewer(ReplayViewer):
    """Several trajectories of the same level side by side, stepped together (see
    gridlock.viz.compare). Same controls and save() as ReplayViewer."""

    def __init__(self, trajs, interval_ms: int = 300):
        trajs = list(trajs)
        self._setup(lambda fig: ComparisonDashboard(trajs, fig), figure_size(len(trajs)), interval_ms)
