"""Runs agent x scenario x seed from a config and writes one JSON line per episode."""

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional


@dataclass
class Trajectory:
    scenario: str
    agent: str
    seed: int
    git_hash: str
    # One dict per step: {t, state, state_hash, actions, R, R_star, events, violations, flags}.
    # `state` is State.to_dict() after the step, so a viewer can draw it without re-simulating.
    steps: list[dict] = field(default_factory=list)
    total_R: float = 0.0
    total_R_star: float = 0.0
    monitor_flags: list[dict] = field(default_factory=list)
    ground_truth_loophole_steps: list[int] = field(default_factory=list)
    grid_rows: list[str] = field(default_factory=list)   # tile layout, so replays don't need the level file
    initial_state: dict = field(default_factory=dict)    # State.to_dict() right after reset

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


StepCallback = Callable[[Trajectory, dict], None]


def load_trajectories(path) -> list[Trajectory]:
    """Read a JSONL file written by run_experiment (one Trajectory per line)."""
    with open(path) as f:
        return [Trajectory.from_dict(json.loads(line)) for line in f if line.strip()]


def run_episode(env, agents: dict, monitor, intent, seed: int,
                on_step: Optional[StepCallback] = None) -> Trajectory:
    """Run one episode and record it.

    `monitor` may be None. `on_step(traj, step)` is called after each step is recorded, for
    example with gridlock.viz.LiveViewer; the runner never imports the viewer itself.
    A step counts as a ground-truth loophole step when `intent.violations` is non-empty.
    """
    obs, _ = env.reset(seed=seed)
    for agent in agents.values():
        agent.reset(env)
    if monitor is not None:
        monitor.reset()

    traj = Trajectory(
        scenario=env.scenario.name,
        agent="+".join(sorted({type(agent).__name__ for agent in agents.values()})),
        seed=seed,
        git_hash=_git_hash(),
        grid_rows=env.grid.to_rows(),
        initial_state=env.get_state().to_dict(),
    )

    while True:
        actions = {aid: agent.act(obs[aid]) for aid, agent in agents.items()}
        obs, rewards, terminated, truncated, info = env.step(actions)
        state, events = env.get_state(), info["events"]
        r_star = {aid: intent.step_score(state, events, aid) for aid in agents}
        violations = list(intent.violations(state, events))
        new_flags = monitor.observe(state.t, state, events, rewards) if monitor is not None else []

        snapshot = state.to_dict()
        step = {
            "t": state.t,
            "state": snapshot,
            "state_hash": _state_hash(snapshot),
            "actions": {aid: action.name for aid, action in actions.items()},
            "R": dict(rewards),
            "R_star": r_star,
            "events": [event.to_dict() for event in events],
            "violations": violations,
            "flags": [asdict(flag) for flag in new_flags],
        }
        traj.steps.append(step)
        traj.total_R += sum(rewards.values())
        traj.total_R_star += sum(r_star.values())
        traj.monitor_flags.extend(step["flags"])
        if violations:
            traj.ground_truth_loophole_steps.append(state.t)
        if on_step is not None:
            on_step(traj, step)
        if all(_flag(terminated, aid) or _flag(truncated, aid) for aid in agents):
            return traj


def run_experiment(config_path: str, out_path: str) -> None:
    raise NotImplementedError


def _flag(flag, agent_id: str) -> bool:
    """Gym-style done flags may be a bool or a per-agent dict. The episode ends once every agent
    is either terminated or truncated, e.g. one at the goal and the other out of time."""
    return bool(flag[agent_id]) if isinstance(flag, dict) else bool(flag)


def _state_hash(snapshot: dict) -> str:
    return hashlib.sha1(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()[:12]


@lru_cache(maxsize=1)
def _git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
