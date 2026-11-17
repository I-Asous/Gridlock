"""Summary tables and plots from experiment results.

    python -m gridlock.eval.report results.jsonl      # print the per-scenario, per-agent table
"""

import argparse
import json
from collections import defaultdict
from itertools import accumulate
from os import PathLike
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from gridlock.eval.metrics import detection_stats
from gridlock.eval.runner import load_trajectories

LogPaths = Union[str, PathLike, Sequence[Union[str, PathLike]], Mapping[str, Union[str, PathLike]]]

SUMMARY_COLUMNS = ("scenario", "agent", "episodes", "mean_R", "mean_R_star", "loophole_rate",
                   "mean_loophole_steps", "recall", "precision", "fpr", "mean_time_to_detect")


def summarize(results_path: str, tolerance: int = 0) -> list[dict]:
    """One row per (scenario, agent) in a JSONL file written by run_experiment, in first-seen order.

    `loophole_rate` is the share of episodes with at least one ground-truth loophole step, and
    `mean_loophole_steps` the mean number of such steps per episode. The detection columns come
    from metrics.detection_stats and are None when there is nothing to measure (e.g. no monitor
    was configured, so nothing was flagged).
    """
    groups = defaultdict(list)
    for traj in load_trajectories(results_path):
        groups[(traj.scenario, traj.agent)].append(traj)
    rows = []
    for (scenario, agent), trajs in groups.items():
        n = len(trajs)
        stats = detection_stats(trajs, tolerance=tolerance)
        rows.append({
            "scenario": scenario, "agent": agent, "episodes": n,
            "mean_R": sum(t.total_R for t in trajs) / n,
            "mean_R_star": sum(t.total_R_star for t in trajs) / n,
            "loophole_rate": sum(bool(t.ground_truth_loophole_steps) for t in trajs) / n,
            "mean_loophole_steps": sum(len(t.ground_truth_loophole_steps) for t in trajs) / n,
            **{key: stats[key] for key in ("recall", "precision", "fpr", "mean_time_to_detect")},
        })
    return rows


def format_summary(rows: list[dict]) -> str:
    """Plain-text table of summarize() rows."""
    def cell(value):
        if value is None:
            return "-"
        return f"{value:.3g}" if isinstance(value, float) else str(value)

    table = [list(SUMMARY_COLUMNS)] + [[cell(row[col]) for col in SUMMARY_COLUMNS] for row in rows]
    widths = [max(len(line[i]) for line in table) for i in range(len(SUMMARY_COLUMNS))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(line, widths)).rstrip() for line in table)


def plot_training_curves(log_paths: LogPaths, out_path: str, window: int = 100) -> None:
    """Plot per-episode R (and R* when it was logged) with a moving average, plus ε, from one or
    more train_log.jsonl files written by gridlock.agents.rl.train.

    `log_paths` is one path, a list of paths (each labelled by its parent directory, e.g.
    runs/q_vase and runs/sarsa_vase), or a {label: path} mapping. With several logs, as for a
    SARSA vs Q-learning comparison, each run gets its own colour: R solid, R* dashed.
    """
    from matplotlib.figure import Figure

    runs = _load_logs(log_paths)
    single = len(runs) == 1
    fig = Figure(figsize=(9, 5.5))
    ax, ax_eps = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": (3, 1)})
    has_r_star = False
    for i, (label, rows) in enumerate(runs.items()):
        episodes = [row["episode"] for row in rows]
        run_color = f"C{i}"
        for key, name, color, style in (("R", "R", "#1f77b4", "-"), ("R_star", "R*", "#2ca02c", "--")):
            if not rows or not all(key in row for row in rows):
                continue
            has_r_star |= key == "R_star"
            color = color if single else run_color
            values = [row[key] for row in rows]
            ax.plot(episodes, values, color=color, alpha=0.2 if single else 0.1, lw=0.8)
            ax.plot(episodes, _moving_average(values, window), color=color, ls=style, lw=1.8,
                    label=f"{name} ({window}-episode mean)" if single else f"{label} {name}")
        ax_eps.plot(episodes, [row["epsilon"] for row in rows], color="#6b7280" if single else run_color,
                    label=None if single else label)
    ax.set_ylabel("episode return")
    ax.set_title("Training: stated reward R vs hidden intent R*" if has_r_star else "Training: stated reward R")
    if any(runs.values()):
        ax.legend(fontsize=8)
    ax_eps.set_ylabel("ε")
    ax_eps.set_xlabel("episode")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def _load_logs(log_paths: LogPaths) -> dict[str, list[dict]]:
    if isinstance(log_paths, (str, PathLike)):
        log_paths = [log_paths]
    if not isinstance(log_paths, Mapping):
        names = [Path(path).parent.name or str(path) for path in log_paths]
        if len(set(names)) < len(names):        # two runs/<same>/ dirs: fall back to full paths
            names = [str(path) for path in log_paths]
        log_paths = dict(zip(names, log_paths))
    runs = {}
    for label, path in log_paths.items():
        with open(path) as f:
            runs[label] = [json.loads(line) for line in f if line.strip()]
    return runs


def _moving_average(values: list[float], window: int) -> list[float]:
    sums = list(accumulate(values, initial=0.0))
    return [(sums[i + 1] - sums[max(0, i + 1 - window)]) / min(i + 1, window) for i in range(len(values))]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m gridlock.eval.report",
                                     description="Summarise a results JSONL from run_experiment.")
    parser.add_argument("results")
    parser.add_argument("--tolerance", type=int, default=0,
                        help="steps a flag may lag a loophole step and still count as correct")
    args = parser.parse_args(argv)
    print(format_summary(summarize(args.results, tolerance=args.tolerance)))


if __name__ == "__main__":
    main()
