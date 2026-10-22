"""Training loop: logs R (and, for analysis only, R*) per episode and checkpoints the agent."""


def train(scenario_name: str, episodes: int, seed: int, out_dir: str, checkpoint_every: int = 500):
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("usage: python -m gridlock.agents.rl.train  (CLI not implemented yet)")
