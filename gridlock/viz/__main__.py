"""Replay recorded episodes.

    python -m gridlock.viz results.jsonl --list                     # list the episodes in the file
    python -m gridlock.viz results.jsonl                            # choose a level, then compare its agents
    python -m gridlock.viz results.jsonl --no-picker                # skip the level screen: first level, first seed
    python -m gridlock.viz results.jsonl --scenario vase --seed 2   # every agent on that level and seed
    python -m gridlock.viz results.jsonl -e 3                       # one episode in detail
    python -m gridlock.viz results.jsonl -e 3 --compare             # every agent on episode 3's level and seed
    python -m gridlock.viz results.jsonl --save cmp.gif             # save instead of opening a window
"""

import argparse
from pathlib import Path

from gridlock.eval.runner import load_trajectories
from gridlock.viz.selection import pick_comparison

__all__ = ["main", "pick_comparison"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m gridlock.viz", description="Replay recorded episodes.")
    parser.add_argument("results", help="JSONL file of trajectories written by gridlock.eval.runner")
    parser.add_argument("-e", "--episode", type=int,
                        help="show this one episode (a line of the file) in detail; without it, every agent "
                             "is shown side by side")
    parser.add_argument("--compare", action="store_true",
                        help="with -e: show every agent on that episode's scenario and seed instead")
    parser.add_argument("--scenario", help="level to compare agents on (default: the first in the file)")
    parser.add_argument("--seed", type=int, help="seed to compare agents on (default: the level's first)")
    parser.add_argument("--list", action="store_true", help="list the episodes in the file and exit")
    parser.add_argument("--no-picker", action="store_true",
                        help="open the comparison straight away instead of the level screen")
    parser.add_argument("--save", metavar="PATH",
                        help="save instead of opening a window: .gif for the whole episode, .png for one frame")
    parser.add_argument("--frame", type=int, help="frame to save with a .png --save (default: last)")
    args = parser.parse_args(argv)

    if not Path(args.results).is_file():
        found = sorted(str(path) for path in Path(".").glob("runs/**/*.jsonl") if path.name != "train_log.jsonl")
        hint = ("; results files under runs/: " + ", ".join(found)) if found else \
            "; write one with python -m gridlock.eval.runner CONFIG.yaml OUT.jsonl"
        parser.error(f"no such file: {args.results}{hint}")
    trajectories = load_trajectories(args.results)
    if not trajectories:
        parser.error(f"no episodes in {args.results}")
    if args.list:
        for i, tr in enumerate(trajectories):
            print(f"{i:>4}  {tr.scenario:<20} {tr.agent:<24} seed={tr.seed:<6} steps={len(tr.steps):<5} "
                  f"R={tr.total_R:+8.2f}  R*={tr.total_R_star:+8.2f}  flags={len(tr.monitor_flags)}")
        return
    if args.episode is not None and not 0 <= args.episode < len(trajectories):
        parser.error(f"--episode must be between 0 and {len(trajectories) - 1}")

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    from gridlock.viz.replay import ComparisonViewer, ReplayViewer

    choosing = args.episode is None and args.scenario is None and args.seed is None
    if choosing and not args.save and not args.no_picker:
        from gridlock.viz.picker import LevelPicker
        LevelPicker(trajectories, source=args.results).show()
        return

    if args.episode is not None and not args.compare:
        viewer = ReplayViewer(trajectories[args.episode])
    else:
        scenario, seed = args.scenario, args.seed
        if args.episode is not None:
            chosen = trajectories[args.episode]
            scenario, seed = chosen.scenario, chosen.seed
        try:
            picked = pick_comparison(trajectories, scenario, seed)
        except ValueError as err:
            parser.error(str(err))
        print(f"{picked[0].scenario}, seed {picked[0].seed}: {', '.join(tr.agent for tr in picked)}")
        viewer = ComparisonViewer(picked)
    if args.save:
        viewer.save(args.save, frame=args.frame)
        print(f"saved {args.save}")
    else:
        viewer.show()


if __name__ == "__main__":
    main()
