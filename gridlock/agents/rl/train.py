"""Training loop: logs R (and, for analysis only, R*) per episode and checkpoints the agent.

    python -m gridlock.agents.rl.train SCENARIO --episodes 5000 --seed 0 --out runs/q_SCENARIO
"""

import argparse
import contextlib
import json
from pathlib import Path
from typing import Optional

from gridlock.agents.base import Agent
from gridlock.agents.rl.q_learning import QLearningAgent
from gridlock.agents.scripted import RandomAgent
from gridlock.env.gridworld import GridWorld
from gridlock.scenarios import load_scenario


def train_agent(env, learner: QLearningAgent, opponents: Optional[dict[str, Agent]] = None,
                episodes: int = 1000, seed: int = 0, out_dir=None, checkpoint_every: int = 500,
                intent=None) -> list[dict]:
    """Train `learner` in `env` next to fixed `opponents` and return one log row per episode.

    The learner only ever sees its own observation and its R. `intent` is optional and only
    logs R* beside R; pass it in from outside gridlock.agents, which must not import the intent
    package. With `out_dir`, rows are written to train_log.jsonl and the learner is saved every
    `checkpoint_every` episodes and at the end.
    """
    agents = {learner.agent_id: learner, **(opponents or {})}
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    rows = []
    with (open(out / "train_log.jsonl", "w") if out is not None else contextlib.nullcontext()) as log:
        for episode in range(episodes):
            row = {"episode": episode, **_run_episode(env, agents, learner, seed + episode, intent)}
            rows.append(row)
            if log is not None:
                log.write(json.dumps(row) + "\n")
                log.flush()
            if out is not None and checkpoint_every and (episode + 1) % checkpoint_every == 0:
                learner.save(out / f"q_ep{episode + 1}.pkl")
    if out is not None:
        learner.save(out / "q_final.pkl")
    return rows


def _run_episode(env, agents: dict[str, Agent], learner: QLearningAgent, seed: int, intent) -> dict:
    lid = learner.agent_id
    obs, _ = env.reset(seed=seed)
    for agent in agents.values():
        agent.reset(env)
    epsilon = learner.epsilon
    total_r = total_r_star = 0.0
    steps = 0
    while True:
        actions = {aid: agent.act(obs[aid]) for aid, agent in agents.items()}
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        done, cut = _for_agent(terminated, lid), _for_agent(truncated, lid)
        learner.update(obs[lid], actions[lid], rewards[lid], next_obs[lid], done)
        total_r += rewards[lid]
        if intent is not None:
            total_r_star += intent.step_score(env.get_state(), info["events"], lid)
        steps += 1
        obs = next_obs
        if done or cut:
            break
    learner.end_episode()
    row = {"R": total_r, "steps": steps, "terminated": done, "epsilon": epsilon, "q_states": len(learner.q)}
    if intent is not None:
        row["R_star"] = total_r_star
    return row


def _for_agent(flag, agent_id: str) -> bool:
    """Gym-style done flags may be a bool or a per-agent dict."""
    return bool(flag[agent_id]) if isinstance(flag, dict) else bool(flag)


def train(scenario_name: str, episodes: int, seed: int, out_dir: str, checkpoint_every: int = 500,
          max_steps: int = 100, **agent_kwargs) -> list[dict]:
    """Train a Q-learner as the scenario's first agent (by sorted id); other agents act at random."""
    scenario = load_scenario(scenario_name)
    env = GridWorld(scenario, max_steps=max_steps)
    learner_id, *other_ids = sorted(scenario.agents)
    learner = QLearningAgent(learner_id, seed=seed, **agent_kwargs)
    opponents = {aid: RandomAgent(aid, seed=seed + i + 1) for i, aid in enumerate(other_ids)}
    return train_agent(env, learner, opponents, episodes, seed, out_dir, checkpoint_every)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m gridlock.agents.rl.train",
                                     description="Train a tabular Q-learner on a scenario, using R only.")
    parser.add_argument("scenario")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="directory for train_log.jsonl and checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    args = parser.parse_args(argv)

    rows = train(args.scenario, args.episodes, args.seed, args.out, args.checkpoint_every, args.max_steps,
                 alpha=args.alpha, gamma=args.gamma, epsilon_decay=args.epsilon_decay)
    if rows:
        tail = rows[-100:]
        print(f"{len(rows)} episodes; mean R over the last {len(tail)}: "
              f"{sum(row['R'] for row in tail) / len(tail):+.2f}; Q-table states: {rows[-1]['q_states']}")


if __name__ == "__main__":
    main()
