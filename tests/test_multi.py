"""The PettingZoo view of a GridWorld, independent learners, and multi-agent experiments."""

import json

import pytest

pytest.importorskip("pettingzoo")

from pettingzoo.test import parallel_api_test  # noqa: E402

from gridlock.agents.rl import QLearningAgent, SARSAAgent  # noqa: E402
from gridlock.agents.rl.multi import train_independent  # noqa: E402
from gridlock.env.grid import Action  # noqa: E402
from gridlock.env.gridworld import GridWorld  # noqa: E402
from gridlock.env.pz_env import GridlockParallelEnv, parallel_env  # noqa: E402
from gridlock.eval.levelcheck import check_level  # noqa: E402
from gridlock.scenarios.loader import load_scenario, parse_scenario  # noqa: E402

TWO_LANES = {"name": "lanes", "layout": ["#####", "#A.G#", "#B.G#", "#####"],
             "legend": {"A": {"agent": "a"}, "B": {"agent": "b"}}, "reward": {"step": -1, "goal": 10}}


def lanes(max_steps=10):
    return GridlockParallelEnv(GridWorld(parse_scenario(TWO_LANES), max_steps=max_steps))


@pytest.mark.parametrize("level", ["vase", "commons"])
def test_passes_the_pettingzoo_parallel_api_test(level):
    parallel_api_test(parallel_env(level), num_cycles=100)


def test_agents_leave_when_they_finish():
    env = lanes()
    obs, infos = env.reset(seed=0)
    assert env.agents == ["a", "b"] and set(obs) == {"a", "b"}
    _, rewards, terminated, truncated, infos = env.step({"a": Action.RIGHT, "b": Action.STAY})
    assert rewards == {"a": -1.0, "b": -1.0} and "events" in infos["a"]
    _, rewards, terminated, _, _ = env.step({"a": Action.RIGHT, "b": Action.RIGHT})
    assert terminated == {"a": True, "b": False} and env.agents == ["b"]
    _, rewards, terminated, _, _ = env.step({"b": Action.RIGHT})
    assert set(rewards) == {"b"} and terminated["b"] and env.agents == []


def test_truncation_ends_every_live_agent():
    env = lanes(max_steps=2)
    env.reset()
    env.step({"a": Action.STAY, "b": Action.STAY})
    _, _, terminated, truncated, _ = env.step({"a": Action.STAY, "b": Action.STAY})
    assert truncated == {"a": True, "b": True} and not any(terminated.values()) and env.agents == []


@pytest.mark.parametrize("cls", [QLearningAgent, SARSAAgent])
def test_independent_learners_both_learn_their_lane(cls, tmp_path):
    env = lanes()
    learners = {aid: cls(aid, alpha=0.5, epsilon_decay=0.98, seed=i) for i, aid in enumerate(env.possible_agents)}
    rows = train_independent(env, learners, episodes=300, seed=0, out_dir=tmp_path, checkpoint_every=0)
    assert set(rows[-1]["R_by_agent"]) == {"a", "b"} and rows[-1]["R"] == sum(rows[-1]["R_by_agent"].values())
    for learner in learners.values():                 # greedy now: both reach their goal
        learner.training = False
    obs, _ = env.reset()
    returns, finished = {"a": 0.0, "b": 0.0}, set()
    while env.agents:
        obs, rewards, terminated, *_ = env.step({aid: learners[aid].act(obs[aid]) for aid in env.agents})
        for aid, reward in rewards.items():
            returns[aid] += reward
        finished |= {aid for aid, done in terminated.items() if done}
    # 2 steps is optimal (R +8). The other agent keeps changing as it learns, so an independent
    # learner may still waste a step or two; anything that reaches the goal within 4 steps passes.
    assert finished == {"a", "b"} and all(r >= 6.0 for r in returns.values())
    assert (tmp_path / f"{cls.name}_a_final.pkl").exists() and (tmp_path / f"{cls.name}_b_final.pkl").exists()
    assert len((tmp_path / "train_log.jsonl").read_text().splitlines()) == 300


def test_independent_training_needs_a_learner_per_agent():
    with pytest.raises(ValueError, match="no learner"):
        train_independent(lanes(), {"a": QLearningAgent("a")}, episodes=1)


def test_commons_is_a_two_agent_loophole_level():
    scenario = load_scenario("commons")
    assert sorted(scenario.agents) == ["a0", "a1"]
    report = check_level(scenario)
    assert report.ok and report.best.violations == ["broke_vase"]


def test_runner_trains_every_agent_with_opponents_learner(tmp_path):
    from gridlock.eval.runner import load_trajectories, run_experiment

    config = tmp_path / "exp.yaml"
    config.write_text(f"scenarios: [commons]\nseeds: [0]\nmax_steps: 40\ntrain_log_dir: {tmp_path / 'runs'}\n"
                      "monitor: {divergence_threshold: 5.0, known_rules: [broke_vase]}\n"
                      "agents:\n  - {name: q2, type: q_learning, opponents: learner, train_episodes: 50}\n")
    run_experiment(str(config), str(tmp_path / "results.jsonl"))
    (traj,) = load_trajectories(tmp_path / "results.jsonl")
    assert traj.agent == "q2" and set(traj.steps[0]["actions"]) == {"a0", "a1"}
    row = json.loads((tmp_path / "runs" / "q2_commons_s0" / "train_log.jsonl").read_text().splitlines()[0])
    assert set(row["R_by_agent"]) == {"a0", "a1"} and set(row["R_star_by_agent"]) == {"a0", "a1"}


@pytest.mark.parametrize("spec", ["{name: x, type: greedy, opponents: learner}",
                                  "{name: x, type: dqn, opponents: learner}"])
def test_opponents_learner_needs_a_tabular_learner(spec, tmp_path):
    from gridlock.eval.runner import run_experiment

    config = tmp_path / "exp.yaml"
    config.write_text(f"scenarios: [commons]\nseeds: [0]\nagents:\n  - {spec}\n")
    with pytest.raises(ValueError, match="tabular"):
        run_experiment(str(config), str(tmp_path / "results.jsonl"))
