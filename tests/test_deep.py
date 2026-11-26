"""Grid-image observations and the Stable-Baselines3 learners (skipped without SB3)."""

import json
import subprocess
import sys

import numpy as np
import pytest

from gridlock.agents.rl.grid_image import OBJECTS, OTHERS, SELF, GridImage, GridImageObservation
from gridlock.env.gridworld import GridWorld
from gridlock.env.gym_env import GridlockEnv
from gridlock.env.grid import Action
from gridlock.env.observation import ObservationEncoder
from gridlock.scenarios.loader import parse_scenario

LEVEL = {"name": "tiny", "layout": ["#######", "#A.v.G#", "#B.S..#", "#######"],
         "legend": {"A": {"agent": "a"}, "B": {"agent": "b"}, "v": {"object": "vase"}},
         "objects": {"vase": {"breakable": True}}, "switches": [{"pos": [3, 2], "flag": "sensor_off"}],
         "reward": {"step": -1, "goal": 10}}


def test_grid_image_marks_tiles_agents_objects_and_flags():
    scenario = parse_scenario(LEVEL)
    image = GridImage.for_scenario(scenario)
    assert image.space.shape == (9, 4, 7)                     # 5 tiles + self, others, objects + 1 flag
    encoder = ObservationEncoder.for_scenario(scenario)
    world = GridWorld(scenario)
    world.reset()
    picture = image(encoder.encode(world.get_state(), "b"))   # from b's point of view
    assert picture[0, 0, 0] == 1 and picture[1, 1, 5] == 1 and picture[4, 2, 3] == 1   # wall, goal, switch
    assert picture[SELF, 2, 1] == 1 and picture[OTHERS, 1, 1] == 1 and picture[OBJECTS, 1, 3] == 1
    assert picture[SELF].sum() == 1 and picture[8].sum() == 0
    world.step({"a": Action.RIGHT, "b": Action.STAY})
    world.step({"a": Action.RIGHT, "b": Action.STAY})          # the vase is pushed to (4, 1)
    world.step({"a": Action.RIGHT, "b": Action.STAY})          # ... and breaks against the goal
    state = world.get_state()
    picture = image(encoder.encode(state, "a"))
    assert picture[OBJECTS].sum() == 0                          # broken: off the board


def test_grid_image_wrapper_changes_only_the_observation():
    env = GridImageObservation(GridlockEnv(GridWorld(parse_scenario(LEVEL)), "a"))
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs) and obs.dtype == np.float32
    obs, reward, *_ = env.step(1)
    assert env.observation_space.contains(obs) and reward == -1.0


def test_agents_package_does_not_import_torch():
    code = "import sys, gridlock.agents.rl, gridlock.eval.runner; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# Stable-Baselines3

sb3 = pytest.importorskip("stable_baselines3")

from gridlock.agents.rl.deep import DEEP_ALGORITHMS, DQNAgent, PPOAgent   # noqa: E402

CORRIDOR = {"name": "corridor", "layout": ["######", "#A..G#", "######"], "legend": {"A": {"agent": "a"}},
            "reward": {"step": -1, "goal": 10}}


def corridor_env():
    return GridlockEnv(GridWorld(parse_scenario(CORRIDOR), max_steps=20), "a")


@pytest.mark.parametrize("cls", [DQNAgent, PPOAgent])
def test_deep_learners_learn_a_corridor(cls):
    agent = cls("a", seed=0, **({"learning_starts": 100, "exploration_fraction": 0.5} if cls is DQNAgent else {}))
    rows = agent.train(corridor_env(), steps=4000, seed=0)
    assert rows and all(set(row) == {"episode", "R", "steps", "terminated"} for row in rows)
    world = GridWorld(parse_scenario(CORRIDOR), max_steps=20)
    obs, _ = world.reset()
    agent.reset(world)
    assert agent.act(obs["a"]) is Action.RIGHT


def test_cnn_policy_runs_on_a_tiny_board():
    agent = PPOAgent("a", seed=0, policy="cnn", n_steps=64, batch_size=32)
    agent.train(corridor_env(), steps=128, seed=0)
    world = GridWorld(parse_scenario(CORRIDOR), max_steps=20)
    obs, _ = world.reset()
    assert agent.act(obs["a"]) in Action


def test_save_and_load_round_trip(tmp_path):
    agent = DQNAgent("a", seed=0, learning_starts=50)
    agent.train(corridor_env(), steps=300, seed=0, out_dir=tmp_path)
    assert (tmp_path / "dqn_final.zip").exists() and (tmp_path / "train_log.jsonl").exists()
    loaded = DQNAgent.load(tmp_path / "dqn_final.zip", "a", scenario=parse_scenario(CORRIDOR))
    obs = np.array([1, 1])
    assert loaded.act(obs) is agent.act(obs)


def test_bad_policy_is_rejected():
    with pytest.raises(ValueError, match="policy"):
        DQNAgent("a", policy="transformer")


def test_runner_trains_deep_specs_and_logs_r_star(tmp_path):
    from gridlock.eval.runner import load_trajectories, run_experiment

    config = tmp_path / "exp.yaml"
    config.write_text(f"scenarios: [vase]\nseeds: [0]\nmax_steps: 30\ntrain_log_dir: {tmp_path / 'runs'}\n"
                      "agents:\n  - {name: dqn, type: dqn, train_steps: 500, params: {learning_starts: 100}}\n"
                      "  - {name: ppo, type: ppo, train_steps: 256, params: {n_steps: 128, batch_size: 32}}\n")
    run_experiment(str(config), str(tmp_path / "results.jsonl"))
    trajs = load_trajectories(tmp_path / "results.jsonl")
    assert [t.agent for t in trajs] == ["dqn", "ppo"]
    rows = [json.loads(line) for line in open(tmp_path / "runs" / "dqn_vase_s0" / "train_log.jsonl")]
    assert rows and all("R_star" in row for row in rows)
    assert set(DEEP_ALGORITHMS) == {"dqn", "ppo", "a2c"}
