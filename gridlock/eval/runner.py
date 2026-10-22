"""Runs agent x scenario x seed from a config and writes one JSON line per episode."""

from dataclasses import dataclass, field


@dataclass
class Trajectory:
    scenario: str
    agent: str
    seed: int
    git_hash: str
    steps: list[dict] = field(default_factory=list)   # {state_hash, actions, R, events}
    total_R: float = 0.0
    total_R_star: float = 0.0
    monitor_flags: list[dict] = field(default_factory=list)
    ground_truth_loophole_steps: list[int] = field(default_factory=list)


def run_episode(env, agents: dict, monitor, intent, seed: int) -> Trajectory:
    raise NotImplementedError


def run_experiment(config_path: str, out_path: str) -> None:
    raise NotImplementedError
