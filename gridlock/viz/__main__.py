"""Replay a recorded episode.

    python -m gridlock.viz results.jsonl --list
    python -m gridlock.viz results.jsonl --episode 3
    python -m gridlock.viz results.jsonl --episode 3 --save ep3.gif
"""

import argparse

from gridlock.eval.runner import load_trajectories


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m gridlock.viz", description="Replay a recorded episode.")
    parser.add_argument("results", help="JSONL file of trajectories written by gridlock.eval.runner")
    parser.add_argument("-e", "--episode", type=int, default=0, help="which line of the file to show (default 0)")
    parser.add_argument("--list", action="store_true", help="list the episodes in the file and exit")
    parser.add_argument("--save", metavar="PATH",
                        help="save instead of opening a window: .gif for the whole episode, .png for one frame")
    parser.add_argument("--frame", type=int, help="frame to save with a .png --save (default: last)")
    args = parser.parse_args(argv)

    trajectories = load_trajectories(args.results)
    if not trajectories:
        parser.error(f"no episodes in {args.results}")
    if args.list:
        for i, tr in enumerate(trajectories):
            print(f"{i:>4}  {tr.scenario:<20} {tr.agent:<24} seed={tr.seed:<6} steps={len(tr.steps):<5} "
                  f"R={tr.total_R:+8.2f}  R*={tr.total_R_star:+8.2f}  flags={len(tr.monitor_flags)}")
        return
    if not 0 <= args.episode < len(trajectories):
        parser.error(f"--episode must be between 0 and {len(trajectories) - 1}")

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    from gridlock.viz.replay import ReplayViewer

    viewer = ReplayViewer(trajectories[args.episode])
    if args.save:
        viewer.save(args.save, frame=args.frame)
        print(f"saved {args.save}")
    else:
        viewer.show()


if __name__ == "__main__":
    main()
