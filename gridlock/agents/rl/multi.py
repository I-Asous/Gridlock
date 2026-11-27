"""Independent learners: several tabular agents learning at the same time on a PettingZoo env.

Each agent is its own QLearningAgent or SARSAAgent with its own Q-table, trained on its own R,
and treats the others as part of the environment (independent Q-learning). The others keep
changing as they learn, so the environment each agent sees isn't stationary. That's the classic
difficulty of multi-agent RL, and part of what a multi-agent level tests.

    env = parallel_env("commons")
    learners = {aid: QLearningAgent(aid, seed=0) for aid in env.possible_agents}
    rows = train_independent(env, learners, episodes=5000, seed=0, out_dir="runs/commons")

Update order per agent is the same as gridlock.agents.rl.train: SARSA chooses its next action
before it updates, Q-learning after.
"""

import contextlib
import json
from pathlib import Path


def train_independent(env, learners: dict, episodes: int = 1000, seed: int = 0, out_dir=None,
                      checkpoint_every: int = 500, intent=None) -> list[dict]:
    """Train `learners` ({agent id: tabular learner}, one per agent in `env.possible_agents`) on
    `env`, a GridlockParallelEnv. Returns one row per episode:

        {episode, R, steps, epsilon, R_by_agent, terminated_by_agent}  (+ R_star, R_star_by_agent)

    R and R_star are summed over agents, so plot_training_curves reads the rows like a single
    learner's. `intent` only logs R* and must come from outside gridlock.agents. With `out_dir`,
    rows go to train_log.jsonl and each learner is saved as `{name}_{agent}_ep{N}.pkl` and
    `{name}_{agent}_final.pkl`."""
    missing = set(env.possible_agents) - set(learners)
    if missing:
        raise ValueError(f"no learner for agents {sorted(missing)}")
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (open(out / "train_log.jsonl", "w") if out is not None else contextlib.nullcontext()) as log:
        for episode in range(episodes):
            row = {"episode": episode, **_run_episode(env, learners, seed if episode == 0 else None, intent)}
            rows.append(row)
            if log is not None:
                log.write(json.dumps(row) + "\n")
                log.flush()
            if out is not None and checkpoint_every and (episode + 1) % checkpoint_every == 0:
                _save(learners, out, f"ep{episode + 1}")
    if out is not None:
        _save(learners, out, "final")
    return rows


def _run_episode(env, learners: dict, seed, intent) -> dict:
    observations, _ = env.reset(seed=seed)
    epsilon = sum(learner.epsilon for learner in learners.values()) / len(learners)
    R = {aid: 0.0 for aid in env.possible_agents}
    R_star = dict(R)
    terminated_by_agent = {aid: False for aid in env.possible_agents}
    actions = {aid: learners[aid].act(observations[aid]) for aid in env.agents}
    steps = 0
    while env.agents:
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        steps += 1
        state = env.world.get_state()
        next_actions = {}
        for aid, action in actions.items():
            learner, done = learners[aid], terminated[aid] or truncated[aid]
            if learner.on_policy:
                next_action = None if terminated[aid] else learner.act(next_obs[aid])
                learner.update(observations[aid], action, rewards[aid], next_obs[aid], terminated[aid], next_action)
            else:
                learner.update(observations[aid], action, rewards[aid], next_obs[aid], terminated[aid])
                next_action = None if done else learner.act(next_obs[aid])
            if not done:
                next_actions[aid] = next_action
            R[aid] += rewards[aid]
            terminated_by_agent[aid] |= terminated[aid]
            if intent is not None:
                R_star[aid] += intent.step_score(state, infos[aid]["events"], aid)
        observations, actions = next_obs, next_actions
    for learner in learners.values():
        learner.end_episode()
    row = {"R": sum(R.values()), "steps": steps, "epsilon": epsilon, "R_by_agent": R,
           "terminated_by_agent": terminated_by_agent}
    if intent is not None:
        row["R_star"] = sum(R_star.values())
        row["R_star_by_agent"] = R_star
    return row


def _save(learners: dict, out: Path, tag: str) -> None:
    for aid, learner in learners.items():
        learner.save(out / f"{learner.name}_{aid}_{tag}.pkl")
