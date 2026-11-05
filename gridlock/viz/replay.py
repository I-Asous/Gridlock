"""Step through a recorded episode: slider, play/pause, arrow keys, and GIF/PNG export."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, Slider

from gridlock.viz.panels import MUTED_COLOR, Dashboard


class ReplayViewer:
    """Interactive viewer for one Trajectory. Keys: ←/→ step, space play/pause, home/end jump."""

    def __init__(self, traj, interval_ms: int = 300):
        self.fig = plt.figure(figsize=(13, 7.5))
        self.dash = Dashboard(traj, self.fig)
        self.last = self.dash.n_frames - 1
        self.frame = 0
        self.playing = False

        slider_ax = self.fig.add_axes((0.08, 0.045, 0.6, 0.03))
        button_ax = self.fig.add_axes((0.74, 0.035, 0.07, 0.05))
        self.slider = Slider(slider_ax, "frame", 0, max(self.last, 1), valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)
        self.button = Button(button_ax, "Play")
        self.button.on_clicked(lambda _event: self.toggle())
        hint = self.fig.text(0.08, 0.012, "← → step    space play/pause    home/end jump",
                             fontsize=8, color=MUTED_COLOR)
        self._controls = (slider_ax, button_ax, hint)

        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.dash.draw(0)

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
        self.button.label.set_text("Pause" if self.playing else "Play")
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
                self.fig.savefig(path, dpi=120)
        finally:
            for artist in self._controls:
                artist.set_visible(True)
            self.dash.draw(self.frame)

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
