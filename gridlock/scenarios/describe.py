"""Plain-language description of a level: what it is, the rules, R, R* and its designed loopholes.

The runner records this in every Trajectory so the viewer can show it without the level file.
It reads the intent section as plain data and never imports gridlock.intent.
"""

from typing import Optional

from gridlock.env.grid import Tile

EVENT_WORDS = {
    "REACHED_GOAL": "reaching the goal", "CHECKPOINT": "entering a checkpoint", "BUMPED": "bumping into something",
    "BLOCKED": "blocking another agent", "PUSHED": "pushing an object", "BROKE": "breaking an object",
    "TOGGLED": "toggling a switch", "PICKED_UP": "picking something up", "MOVED": "moving",
    "TRAPPED": "stepping on a trap",
}

GENERAL_RULES = [
    "Agents act at the same time (UP, DOWN, LEFT, RIGHT, STAY, INTERACT). Walls stop them.",
    "Two agents can't enter the same cell or swap cells. Reaching G ends that agent's episode.",
]


def describe_scenario(scenario, max_steps: Optional[int] = None) -> dict:
    """{"name", "description", "rules", "reward", "intent", "loopholes"}: strings and lists of
    strings, JSON-friendly. Works on partial scenario objects (missing parts are left out)."""
    rules = list(GENERAL_RULES)
    if max_steps is not None:
        rules[-1] += f" Episodes are cut off after {max_steps} steps."
    for oid, props in sorted((getattr(scenario, "object_props", None) or {}).items()):
        rules.append(f"{oid}: {_object_rule(props)}")
    trap_off = set(getattr(scenario, "trap_off_flags", None) or ())
    for (x, y), flag in sorted((getattr(scenario, "switches", None) or {}).items()):
        effect = " While it is on, traps do nothing." if flag in trap_off else ""
        rules.append(f"Switch at ({x}, {y}): INTERACT on it toggles '{flag}'.{effect}")
    grid = getattr(scenario, "grid", None)
    if grid is not None and grid.cells(Tile.CHECKPOINT):
        rules.append("C: a bonus point. Entering it counts every time, not just the first.")
    if grid is not None and grid.cells(Tile.TRAP):
        rules.append("^: a trap. Stepping on it sends the agent back to where it started.")

    intent = getattr(scenario, "intent_config", None) or {}
    loopholes = []
    for entry in getattr(scenario, "designed_loopholes", None) or []:
        text = " ".join(str(entry.get("description", "")).split())
        loopholes.append(f"{entry.get('name', 'loophole')}: {text}" if text else str(entry.get("name", "loophole")))

    return {
        "name": getattr(scenario, "name", ""),
        "description": " ".join(str(getattr(scenario, "description", "") or "").split()),
        "rules": rules,
        "reward": _joined(_reward_lines(getattr(getattr(scenario, "reward_fn", None), "config", None) or {})),
        "intent": _joined(_reward_lines(intent.get("base") or {})) + [_forbid_line(rule)
                                                                      for rule in intent.get("forbid") or []],
        "loopholes": loopholes,
    }


def _joined(lines: list[str]) -> list[str]:
    return ["; ".join(lines)] if lines else []


def _object_rule(props: dict) -> str:
    pushable, breakable = props.get("pushable"), props.get("breakable")
    if pushable and breakable:
        return "pushing moves it one cell onto free floor; if it can't move, it breaks."
    if pushable:
        return "pushing moves it one cell onto free floor; if it can't move, nothing happens."
    if breakable:
        return "walking into it breaks it."
    return "a fixed obstacle."


def _reward_lines(config: dict) -> list[str]:
    lines = []
    for key, value in config.items():
        if key == "step":
            lines.append(f"{value:+g} per step")
        else:
            kind = "REACHED_GOAL" if key == "goal" else key.upper()
            lines.append(f"{value:+g} for {EVENT_WORDS.get(kind, kind.lower())}")
    return lines


def _forbid_line(rule: dict) -> str:
    penalty = rule.get("penalty", -10)
    if "area" in rule:
        (x0, y0), (x1, y1) = rule["area"]
        what = f"entering any cell in x {min(x0, x1)}..{max(x0, x1)}, y {min(y0, y1)}..{max(y0, y1)}"
    elif "cells" in rule:
        cells = ", ".join(f"({x}, {y})" for x, y in rule["cells"])
        what = f"entering {cells}"
    else:
        what = str(rule.get("kind", "")).upper()
        if rule.get("target"):
            what += f" {rule['target']}"
        if rule.get("payload"):
            what += " " + ", ".join(f"{k}={v}" for k, v in rule["payload"].items())
    return f"{rule.get('name', 'rule')}: {penalty:+g} for {what}"
