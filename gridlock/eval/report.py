"""Summary tables and plots from experiment results."""

import json
from itertools import accumulate


def summarize(results_path: str):
    raise NotImplementedError


def plot_training_curves(log_path: str, out_path: str, window: int = 100) -> None:
    """Plot per-episode R (and R* when it was logged) with a moving average, plus ε, from a
    train_log.jsonl written by gridlock.agents.rl.train."""
    from matplotlib.figure import Figure

    with open(log_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    episodes = [row["episode"] for row in rows]

    fig = Figure(figsize=(9, 5.5))
    ax, ax_eps = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": (3, 1)})
    for key, label, color, style in (("R", "R", "#1f77b4", "-"), ("R_star", "R*", "#2ca02c", "--")):
        if rows and all(key in row for row in rows):
            values = [row[key] for row in rows]
            ax.plot(episodes, values, color=color, alpha=0.2, lw=0.8)
            ax.plot(episodes, _moving_average(values, window), color=color, ls=style, lw=1.8,
                    label=f"{label} ({window}-episode mean)")
    ax.set_ylabel("episode return")
    ax.set_title("Training: stated reward R vs hidden intent R*" if rows and "R_star" in rows[0]
                 else "Training: stated reward R")
    if rows:
        ax.legend(fontsize=8)
    ax_eps.plot(episodes, [row["epsilon"] for row in rows], color="#6b7280")
    ax_eps.set_ylabel("ε")
    ax_eps.set_xlabel("episode")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def _moving_average(values: list[float], window: int) -> list[float]:
    sums = list(accumulate(values, initial=0.0))
    return [(sums[i + 1] - sums[max(0, i + 1 - window)]) / min(i + 1, window) for i in range(len(values))]
