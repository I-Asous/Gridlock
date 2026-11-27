"""Loads level YAML files into Scenario objects.

A level file looks like this (see levels/vase.yaml for a full example):

    name: vase
    description: one line on what the level tests
    layout:                       # tiles (. # G C S ^, see gridlock.env.grid.Tile) plus legend markers
      - "#######"
      - "#A.v.G#"
      - "#######"
    legend:                       # every non-tile character in the layout, each used exactly once
      A: {agent: a0}
      v: {object: vase}           # optional "tile": the tile under the marker (default ".")
    objects:                      # optional per-object properties
      vase: {pushable: true, breakable: true}
    switches:                     # optional: INTERACT on an S tile toggles this state flag
      - {pos: [3, 1], flag: sensor_off}
      - {pos: [1, 2], flag: traps_off, disables: traps}   # while this flag is on, traps do nothing
    flags: []                     # extra state flags agents observe (switch flags are added)
    reward: {step: -1, goal: 10}  # R, see gridlock.rules.reward.RewardFn
    intent:                       # R*, see gridlock.intent.intent.IntentFn; read only by monitor / eval
      base: {step: -1, goal: 10}
      forbid:
        - {name: broke_vase, kind: BROKE, target: vase, penalty: -20}
    loopholes:                    # the designed loopholes; names match intent rules
      - {name: broke_vase, description: "..."}
    control: false                # true for a level with no profitable loophole (checks false positives)
    max_steps: 300                # optional: this level's step limit, overriding the experiment's (for big levels)

`python -m gridlock.eval.levelcheck` checks that a level's loopholes really pay off under R.

This module never imports gridlock.intent: it passes the intent section through untouched.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from gridlock.env.events import Pos
from gridlock.env.grid import Grid, Tile
from gridlock.rules.reward import RewardFn

LEVELS_DIR = Path(__file__).parent / "levels"

TILE_CHARS = {tile.value for tile in Tile}
OBJECT_DEFAULTS = {"pushable": True, "breakable": False}
LEVEL_KEYS = {"name", "description", "layout", "legend", "objects", "switches", "flags", "reward", "intent",
              "loopholes", "control", "max_steps"}
SWITCH_KEYS = {"pos", "flag", "disables"}


@dataclass
class Scenario:
    name: str
    grid: Grid
    agents: dict[str, tuple[int, int]]      # agent id -> start position
    reward_fn: RewardFn
    intent_config: dict                     # read only by monitor / eval
    designed_loopholes: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)   # state flags the level can set, e.g. "sensor_off"; observed
    object_props: dict[str, dict] = field(default_factory=dict)   # object id -> {"pushable", "breakable"}
    switches: dict[Pos, str] = field(default_factory=dict)       # switch tile position -> flag it toggles
    description: str = ""
    control: bool = False                   # no profitable loophole by design
    trap_off_flags: list[str] = field(default_factory=list)   # flags that, while on, disable every trap
    max_steps: Optional[int] = None         # the level's own step limit; None: the caller's default


def available_levels() -> list[str]:
    return sorted(path.stem for path in LEVELS_DIR.glob("*.yaml"))


def load_scenario(name: Union[str, Path]) -> Scenario:
    """Load a level by name (a file in LEVELS_DIR) or by path to a .yaml file."""
    path = Path(name)
    if path.suffix not in (".yaml", ".yml"):
        path = LEVELS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no level {str(name)!r}; available levels: {', '.join(available_levels()) or 'none'}")
    with open(path) as f:
        spec = yaml.safe_load(f)
    return parse_scenario(spec, default_name=path.stem)


def parse_scenario(spec: dict[str, Any], default_name: str = "level") -> Scenario:
    """Build a Scenario from an already-parsed level dict (the YAML format above)."""
    if not isinstance(spec, dict):
        raise ValueError("a level must be a mapping")
    unknown = set(spec) - LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown level keys: {sorted(unknown)}")
    name = spec.get("name", default_name)
    layout = spec.get("layout")
    if not layout or not all(isinstance(row, str) for row in layout):
        raise ValueError(f"level {name!r}: layout must be a non-empty list of strings")

    legend = spec.get("legend") or {}
    clashes = sorted(ch for ch in legend if ch in TILE_CHARS or len(ch) != 1)
    if clashes:
        raise ValueError(f"level {name!r}: legend markers must be single characters that aren't tiles "
                         f"({''.join(sorted(TILE_CHARS))}); got {clashes}")
    agents: dict[str, Pos] = {}
    objects: dict[str, Pos] = {}
    seen: set[str] = set()
    rows = []
    for y, row in enumerate(layout):
        cells = []
        for x, ch in enumerate(row):
            if ch in TILE_CHARS:
                cells.append(ch)
                continue
            entry = legend.get(ch)
            if not isinstance(entry, dict):
                raise ValueError(f"level {name!r}: character {ch!r} at ({x}, {y}) is neither a tile nor in the legend")
            if ch in seen:
                raise ValueError(f"level {name!r}: legend marker {ch!r} is used more than once")
            seen.add(ch)
            tile = entry.get("tile", Tile.FLOOR.value)
            if tile not in TILE_CHARS or tile == Tile.WALL.value:
                raise ValueError(f"level {name!r}: marker {ch!r} has invalid tile {tile!r}")
            cells.append(tile)
            if "agent" in entry:
                agents[str(entry["agent"])] = (x, y)
            elif "object" in entry:
                objects[str(entry["object"])] = (x, y)
            else:
                raise ValueError(f"level {name!r}: legend entry {ch!r} needs an 'agent' or 'object' id")
        rows.append("".join(cells))
    if not agents:
        raise ValueError(f"level {name!r}: no agents in the layout")
    grid = Grid.from_rows(rows, objects=objects)
    for aid, pos in agents.items():
        if grid.tile_at(pos) is Tile.GOAL:
            raise ValueError(f"level {name!r}: agent {aid!r} starts on a goal")

    object_props = {oid: dict(OBJECT_DEFAULTS) for oid in objects}
    for oid, props in (spec.get("objects") or {}).items():
        if oid not in objects:
            raise ValueError(f"level {name!r}: properties given for {oid!r}, which is not in the layout")
        bad = set(props or {}) - set(OBJECT_DEFAULTS)
        if bad:
            raise ValueError(f"level {name!r}: unknown properties for {oid!r}: {sorted(bad)}")
        object_props[oid].update({key: bool(value) for key, value in (props or {}).items()})

    switches: dict[Pos, str] = {}
    trap_off_flags: list[str] = []
    for entry in spec.get("switches") or []:
        pos = tuple(entry["pos"])
        if len(pos) != 2 or not grid.in_bounds(pos) or grid.tile_at(pos) is not Tile.SWITCH:
            raise ValueError(f"level {name!r}: switch at {list(pos)} is not on an S tile")
        bad = set(entry) - SWITCH_KEYS
        if bad:
            raise ValueError(f"level {name!r}: unknown keys for the switch at {list(pos)}: {sorted(bad)}")
        switches[pos] = str(entry["flag"])
        disables = entry.get("disables")
        if disables is not None:
            if disables != "traps":
                raise ValueError(f"level {name!r}: switch at {list(pos)} can only disable 'traps', got {disables!r}")
            if switches[pos] not in trap_off_flags:
                trap_off_flags.append(switches[pos])

    flags = list(dict.fromkeys([*(spec.get("flags") or []), *switches.values()]))
    return Scenario(
        name=name,
        grid=grid,
        agents=agents,
        reward_fn=RewardFn(spec.get("reward") or {}),
        intent_config=dict(spec.get("intent") or {}),
        designed_loopholes=list(spec.get("loopholes") or []),
        flags=flags,
        object_props=object_props,
        switches=switches,
        description=spec.get("description", ""),
        control=bool(spec.get("control", False)),
        trap_off_flags=trap_off_flags,
        max_steps=_max_steps(spec.get("max_steps"), name),
    )


def _max_steps(value, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"level {name!r}: max_steps must be a positive integer, got {value!r}")
    return value
