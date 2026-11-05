"""Training loop on the Gymnasium API: logs R (and, for analysis only, R*) per episode and
checkpoints the agent.

    python -m gridlock.agents.rl.train SCENARIO --episodes 5000 --seed 0 --out runs/q_SCENARIO
"""

import argparse
import contextlib
import json
from pathlib import Path

import gymnasium
from gymnasium.wrappers import RecordEpisodeStatistics

from gridlock.agents.rl.q_learning import QLearningAgent
from gridlock.env.gym_env import ENV_ID


def train_agent(env: gymnasium.Env, learner: QLearningAgent, episodes: int = 1000, seed: int = 0,
                out_dir=None, checkpoint_every: int = 500, intent=None) -> list[dict]:
    """Train `learner` on a Gymnasium environment (normally a GridlockEnv) and return one log row
    per episode.

    Only the first reset is seeded; later episodes continue the environment's RNG, as Gymnasium
    recommends. `intent` is optional and only logs R* beside R. It reads the underlying State
    through `env.unwrapped.state`, and must be passed in from outside gridlock.agents, which may
    not import the intent package. With `out_dir`, rows are written to train_log.jsonl and the
    learner is saved every `checkpoint_every` episodes and at the end.
    """
    env = RecordEpisodeStatistics(env)
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    rows = []
    with (open(out / "train_log.jsonl", "w") if out is not None else contextlib.nullcontext()) as log:
        for episode in range(episodes):
            row = {"episode": episode, **_run_episode(env, learner, seed if episode == 0 else None, intent)}
            rows.append(row)
            if log is not None:
                log.write(json.dumps(row) + "\n")
                log.flush()
            if out is not None and checkpoint_every and (episode + 1) % checkpoint_every == 0:
                learner.save(out / f"q_ep{episode + 1}.pkl")
    if out is not None:
        learner.save(out / "q_final.pkl")
    return rows


def _run_episode(env: gymnasium.Env, learner: QLearningAgent, seed, intent) -> dict:
    obs, info = env.reset(seed=seed)
    epsilon = learner.epsilon
    r_star = 0.0
    while True:
        action = learner.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        learner.update(obs, action, reward, next_obs, terminated)
        if intent is not None:
            base = env.unwrapped
            r_star += intent.step_score(base.state, info["events"], base.learner_id)
        obs = next_obs
        if terminated or truncated:
            break
    learner.end_episode()
    stats = info["episode"]          # added by RecordEpisodeStatistics on the last step
    row = {"R": float(stats["r"]), "steps": int(stats["l"]), "terminated": bool(terminated),
           "epsilon": epsilon, "q_states": len(learner.q)}
    if intent is not None:
        row["R_star"] = r_star
    return row


def train(scenario_name: str, episodes: int, seed: int, out_dir: str, checkpoint_every: int = 500,
          max_steps: int = 100, **agent_kwargs) -> list[dict]:
    """Train a Q-learner as the scenario's first agent (by sorted id); other agents act at random."""
    env = gymnasium.make(ENV_ID, scenario=scenario_name, max_steps=max_steps, seed=seed)
    learner = QLearningAgent(env.unwrapped.learner_id, seed=seed, **agent_kwargs)
    return train_agent(env, learner, episodes, seed, out_dir, checkpoint_every)


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
