"""Watch an episode while it runs.

    viewer = LiveViewer(pause=0.2)
    traj = run_episode(env, agents, monitor, intent, seed=0, on_step=viewer)
    viewer.show()   # keep the window open afterwards
"""

import matplotlib.pyplot as plt

from gridlock.viz.panels import Dashboard


class LiveViewer:
    """An on_step callback for run_episode that redraws the dashboard after every step.

    Reusing one viewer across episodes reuses its window. Closing the window stops drawing
    and lets the remaining steps run at full speed.
    """

    def __init__(self, pause: float = 0.2):
        self.pause = pause
        self.fig = None
        self.dash = None

    def __call__(self, traj, step: dict) -> None:
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(13, 7))
        elif not plt.fignum_exists(self.fig.number):
            return
        if self.dash is None or self.dash.traj is not traj:
            self.fig.clear()
            self.dash = Dashboard(traj, self.fig)
        self.dash.draw(self.dash.n_frames - 1)
        plt.pause(self.pause)

    def show(self) -> None:
        """Block until the window is closed."""
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.ioff()
            plt.show()
