"""Step through a recorded episode: slider, play/pause, arrow keys, and GIF/PNG export."""

from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, Slider

from gridlock.viz import theme
from gridlock.viz.compare import ComparisonDashboard, figure_size
from gridlock.viz.panels import Dashboard
from gridlock.viz.theme import ACCENT, ACCENT_DARK, GRIDLINE, INK, MUTED, Layout, themed


class ReplayViewer:
    """Interactive viewer for one Trajectory. Keys: ←/→ step, space play/pause, home/end jump.

    With `on_back`, the control bar gets a "← levels" button (also esc) that calls it and closes
    this window; the level picker uses it to come back."""

    def __init__(self, traj, interval_ms: int = 300, on_back: Optional[Callable[[], None]] = None):
        self._setup(lambda fig: Dashboard(traj, fig), (14, 8), interval_ms, on_back)

    def _setup(self, make_dashboard, figsize, interval_ms: int, on_back=None) -> None:
        self.on_back = on_back
        with themed():
            self.fig = plt.figure(figsize=figsize)
            if self.fig.canvas.manager is not None:
                self.fig.canvas.manager.set_window_title("Gridlock replay")
            self.fig._gridlock_owner = self           # callbacks hold only weak refs to bound methods
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
        """A bar along the bottom: back button (with on_back), play button, frame slider and key
        hints. Hidden when saving."""
        L = Layout(self.fig)
        m, y, h = 0.25, L.h - 0.6, 0.48
        card = L.card(m, y, L.w - 2 * m, h)
        extra = ()
        x = m + 0.12
        if self.on_back is not None:
            back_ax = L.axes(x, y + 0.08, 1.0, h - 0.16)
            self.back_button = Button(back_ax, "←  levels", color="white", hovercolor=GRIDLINE)
            self.back_button.label.set_color(INK)
            for spine in back_ax.spines.values():          # the theme hides top and right spines
                spine.set_visible(True)
                spine.set_color(INK)
            self.back_button.on_clicked(lambda _event: self.back())
            extra, x = (back_ax,), x + 1.12
        button_ax = L.axes(x, y + 0.08, 1.0, h - 0.16)
        slider_ax = L.axes(x + 1.28, y + 0.14, L.w - 2 * m - 5.4 - (x - m - 0.12) * 2, h - 0.28)
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
        hints = "← →  step   /   space  play·pause   /   home end  jump"
        if self.on_back is not None:
            hints += "   /   esc  levels"
        hint = L.text(L.w - m - 0.2, y + h / 2, hints, fontsize=7.5, color=MUTED, ha="right", va="center")
        self._controls = (*card, *extra, slider_ax, button_ax, hint)

    def back(self) -> None:
        """Go back to the level picker: call on_back, then close this window."""
        if self.on_back is None:
            return
        self.timer.stop()
        self.on_back()
        plt.close(self.fig)

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
        elif event.key == "escape":
            self.back()
        elif event.key in jumps:
            self.go(jumps[event.key])


class ComparisonViewer(ReplayViewer):
    """Several trajectories of the same level side by side, stepped together (see
    gridlock.viz.compare). Same controls and save() as ReplayViewer."""

    def __init__(self, trajs, interval_ms: int = 300, on_back: Optional[Callable[[], None]] = None):
        trajs = list(trajs)
        self._setup(lambda fig: ComparisonDashboard(trajs, fig), figure_size(len(trajs)), interval_ms, on_back)
