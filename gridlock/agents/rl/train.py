"""Training loop on the Gymnasium API: logs R (and, for analysis only, R*) per episode and
checkpoints the agent. Works for both tabular learners: Q-learning (off-policy) and SARSA
(on-policy).

    python -m gridlock.agents.rl.train SCENARIO --algo q     --episodes 5000 --seed 0 --out runs/q_SCENARIO
    python -m gridlock.agents.rl.train SCENARIO --algo sarsa --episodes 5000 --seed 0 --out runs/sarsa_SCENARIO
"""

import argparse
import contextlib
import json
from pathlib import Path

import gymnasium
from gymnasium.wrappers import RecordEpisodeStatistics

from gridlock.agents.rl import ALGORITHMS
from gridlock.agents.rl.q_learning import QLearningAgent
from gridlock.env.gym_env import ENV_ID


def train_agent(env: gymnasium.Env, learner: QLearningAgent, episodes: int = 1000, seed: int = 0,
                out_dir=None, checkpoint_every: int = 500, intent=None) -> list[dict]:
    """Train `learner` (a QLearningAgent or SARSAAgent) on a Gymnasium environment (normally a
    GridlockEnv) and return one log row per episode.

    Only the first reset is seeded; later episodes continue the environment's RNG, as Gymnasium
    recommends. `intent` is optional and only logs R* beside R. It reads the underlying State
    through `env.unwrapped.state`, and must be passed in from outside gridlock.agents, which may
    not import the intent package. With `out_dir`, rows are written to train_log.jsonl and the
    learner is saved every `checkpoint_every` episodes and at the end, as
    `{learner.name}_ep{N}.pkl` and `{learner.name}_final.pkl`.
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
                learner.save(out / f"{learner.name}_ep{episode + 1}.pkl")
    if out is not None:
        learner.save(out / f"{learner.name}_final.pkl")
    return rows


def _run_episode(env: gymnasium.Env, learner: QLearningAgent, seed, intent) -> dict:
    """One episode. The action for each step is chosen at the end of the previous one, so an
    on-policy learner (SARSA) can back up the action it really takes next. An off-policy learner
    (Q-learning) updates first and then chooses, as in the textbook algorithm. Neither chooses an
    action after the episode is over, except SARSA after a truncation, whose backup still needs a'."""
    obs, info = env.reset(seed=seed)
    epsilon = learner.epsilon
    r_star = 0.0
    action = learner.act(obs)
    while True:
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if learner.on_policy:
            next_action = None if terminated else learner.act(next_obs)
            learner.update(obs, action, reward, next_obs, terminated, next_action)
        else:
            learner.update(obs, action, reward, next_obs, terminated)
            next_action = None if done else learner.act(next_obs)
        if intent is not None:
            base = env.unwrapped
            r_star += intent.step_score(base.state, info["events"], base.learner_id)
        if done:
            break
        obs, action = next_obs, next_action
    learner.end_episode()
    stats = info["episode"]          # added by RecordEpisodeStatistics on the last step
    row = {"R": float(stats["r"]), "steps": int(stats["l"]), "terminated": bool(terminated),
           "epsilon": epsilon, "q_states": len(learner.q)}
    if intent is not None:
        row["R_star"] = r_star
    return row


def train(scenario_name: str, episodes: int, seed: int, out_dir: str, checkpoint_every: int = 500,
          max_steps: int = 100, algo: str = "q", **agent_kwargs) -> list[dict]:
    """Train a tabular learner (`algo` is a key of ALGORITHMS: "q" or "sarsa") as the scenario's
    first agent (by sorted id); other agents act at random."""
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algo {algo!r}; expected one of {sorted(ALGORITHMS)}")
    env = gymnasium.make(ENV_ID, scenario=scenario_name, max_steps=max_steps, seed=seed)
    learner = ALGORITHMS[algo](env.unwrapped.learner_id, seed=seed, **agent_kwargs)
    return train_agent(env, learner, episodes, seed, out_dir, checkpoint_every)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m gridlock.agents.rl.train",
                                     description="Train a tabular Q-learning or SARSA agent on a scenario, "
                                                 "using R only.")
    parser.add_argument("scenario")
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="q",
                        help="q: off-policy Q-learning (default); sarsa: on-policy SARSA")
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
                 algo=args.algo, alpha=args.alpha, gamma=args.gamma, epsilon_decay=args.epsilon_decay)
    if rows:
        tail = rows[-100:]
        print(f"{len(rows)} episodes; mean R over the last {len(tail)}: "
              f"{sum(row['R'] for row in tail) / len(tail):+.2f}; Q-table states: {rows[-1]['q_states']}")


if __name__ == "__main__":
    main()
