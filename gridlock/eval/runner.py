"""Runs agent x scenario x seed from a config and writes one JSON line per episode.

    python -m gridlock.eval.runner experiments/sarsa_vs_q.yaml results.jsonl
"""

import argparse
import hashlib
import inspect
import json
import subprocess
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import yaml

from gridlock.agents.astar import AStarAgent
from gridlock.agents.csp import CSPAgent
from gridlock.agents.rl import ALGORITHMS
from gridlock.agents.rl.train import train_agent
from gridlock.agents.scripted import GreedyAgent, RandomAgent
from gridlock.env.gridworld import GridWorld
from gridlock.env.gym_env import GridlockEnv
from gridlock.intent import IntentFn
from gridlock.monitor import Monitor
from gridlock.scenarios import load_scenario
from gridlock.scenarios.describe import describe_scenario

# Config `type` -> agent class. Learners (ALGORITHMS) are trained before they are evaluated.
AGENT_TYPES = {"random": RandomAgent, "greedy": GreedyAgent, "astar": AStarAgent, "csp": CSPAgent,
               "q_learning": ALGORITHMS["q"], "sarsa": ALGORITHMS["sarsa"]}
LEARNERS = {"q_learning", "sarsa"}


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
    level: dict = field(default_factory=dict)            # describe_scenario(): description, rules, R, R*, loopholes

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
        level=describe_scenario(env.scenario, getattr(env, "max_steps", None)),
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
    """Run every agent spec on every scenario for every seed and write one Trajectory per line.

    The config is YAML:

        scenarios: [vase]
        seeds: [0, 1, 2]
        max_steps: 100                  # optional, default 100
        train_log_dir: runs/sarsa_vs_q  # optional: keep each learner's train_log.jsonl + checkpoints
        monitor: {divergence_threshold: 0.5}   # optional; omit to record no flags
        agents:
          - {name: q,     type: q_learning, train_episodes: 5000, params: {alpha: 0.1}}
          - {name: sarsa, type: sarsa,      train_episodes: 5000, params: {alpha: 0.1}}
          - {name: greedy, type: greedy}

    Each spec controls the scenario's first agent (by sorted id). Every other agent is an
    `opponents` type (default "random", seeded per run). `type` is a key of AGENT_TYPES and
    `params` go to its constructor. Learners are trained on R alone for `train_episodes` on a
    fresh world (R* is logged beside R when `train_log_dir` is set), then evaluated greedily for
    one episode with the same seed. The Trajectory's `agent` field is the spec's `name`, so
    results group by spec rather than by class.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    max_steps = config.get("max_steps", 100)
    train_dir = Path(config["train_log_dir"]) if config.get("train_log_dir") else None
    specs = config["agents"]
    for spec in specs:
        for key in ("type", *(("opponents",) if "opponents" in spec else ())):
            kind = spec[key]
            if kind not in AGENT_TYPES or (key == "opponents" and kind in LEARNERS):
                raise ValueError(f"agent spec {spec.get('name')!r}: unsupported {key} {kind!r}")
    if len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("agent spec names must be unique")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as out:
        for scenario_name in config["scenarios"]:
            scenario = load_scenario(scenario_name)
            for spec in specs:
                for seed in config["seeds"]:
                    traj = _run_spec(scenario, spec, seed, max_steps, config.get("monitor"), train_dir)
                    out.write(traj.to_json() + "\n")
                    out.flush()


def _run_spec(scenario, spec: dict, seed: int, max_steps: int, monitor_config: Optional[dict],
              train_dir: Optional[Path]) -> Trajectory:
    agent_ids = sorted(scenario.agents)
    learner_id, other_ids = agent_ids[0], agent_ids[1:]
    intent = IntentFn(scenario.intent_config)

    def opponents() -> dict:
        kind = spec.get("opponents", "random")
        return {aid: _build(kind, aid, {}, seed + i + 1) for i, aid in enumerate(other_ids)}

    learner = _build(spec["type"], learner_id, spec.get("params", {}), seed)
    if spec["type"] in LEARNERS:
        env = GridlockEnv(GridWorld(scenario, max_steps=max_steps), learner_id, opponents())
        out_dir = train_dir / f"{spec['name']}_{scenario.name}_s{seed}" if train_dir is not None else None
        train_agent(env, learner, spec.get("train_episodes", 1000), seed, out_dir=out_dir,
                    intent=intent if out_dir is not None else None)
        learner.training = False

    world = GridWorld(scenario, max_steps=max_steps)
    monitor = Monitor(intent, **monitor_config) if monitor_config else None
    traj = run_episode(world, {learner_id: learner, **opponents()}, monitor, intent, seed)
    traj.agent = spec["name"]
    return traj


def _build(kind: str, agent_id: str, params: dict, seed: int):
    cls = AGENT_TYPES[kind]
    if "seed" in inspect.signature(cls).parameters and "seed" not in params:
        params = {**params, "seed": seed}
    return cls(agent_id, **params)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m gridlock.eval.runner",
                                     description="Run an experiment config and write trajectories as JSONL.")
    parser.add_argument("config")
    parser.add_argument("out")
    args = parser.parse_args(argv)
    run_experiment(args.config, args.out)


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


if __name__ == "__main__":
    main()
