"""Loads level YAML files into Scenario objects."""

from dataclasses import dataclass, field
from pathlib import Path

from gridlock.env.grid import Grid
from gridlock.rules.reward import RewardFn

LEVELS_DIR = Path(__file__).parent / "levels"


@dataclass
class Scenario:
    name: str
    grid: Grid
    agents: dict[str, tuple[int, int]]      # agent id -> start position
    reward_fn: RewardFn
    intent_config: dict                     # read only by monitor / eval
    designed_loopholes: list[dict] = field(default_factory=list)


def load_scenario(name: str) -> Scenario:
    raise NotImplementedError
