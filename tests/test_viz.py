"""Rendering, trajectory recording and the replay viewer, driven by a tiny fake environment."""

import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest

from gridlock.agents.base import Agent
from gridlock.env.events import Event, EventKind
from gridlock.env.grid import Action, Grid, Tile
from gridlock.env.gridworld import State, object_positions
from gridlock.eval.runner import Trajectory, load_trajectories, run_episode
from gridlock.monitor.monitor import Flag

ROWS = ["#######", "#.....#", "#.S..G#", "#######"]


class FakeEnv:
    """Agents walk right; agent b breaks the vase at t=2. Ends after 4 steps."""

    def __init__(self):
        self.grid = Grid.from_rows(ROWS)
        self.scenario = SimpleNamespace(name="fake")
        self.state = None

    def reset(self, seed=None):
        self.state = State(t=0, agents=(("a", (1, 1)), ("b", (1, 2))), objects=(("vase", (3, 2)),))
        return {"a": None, "b": None}, {}

    def get_state(self):
        return self.state

    def step(self, actions):
        t = self.state.t + 1
        agents, events = [], []
        for aid, (x, y) in self.state.agents:
            dx, dy = actions[aid].value or (0, 0)
            if self.grid.is_walkable((x + dx, y + dy)):
                x, y = x + dx, y + dy
                events.append(Event(t, aid, EventKind.MOVED, (x, y)))
            agents.append((aid, (x, y)))
        objects = self.state.objects
        if t == 2:
            events.append(Event(t, "b", EventKind.BROKE, (3, 2), target="vase"))
            objects = (("vase", None),)
        self.state = State(t, tuple(agents), objects, frozenset({"sensor_off"}) if t >= 3 else frozenset())
        done = t >= 4
        return {aid: None for aid in actions}, {aid: -1.0 for aid in actions}, {aid: done for aid in actions}, False, {"events": events}


class RightAgent(Agent):
    def act(self, obs):
        return Action.RIGHT


class FakeIntent:
    def step_score(self, state, events, agent_id):
        return -5.0 if any(e.kind is EventKind.BROKE and e.agent_id == agent_id for e in events) else 0.0

    def violations(self, state, events):
        return ["no_break_vase"] if any(e.kind is EventKind.BROKE for e in events) else []


class FakeMonitor:
    def reset(self):
        pass

    def observe(self, t, state, events, rewards):
        return [Flag(t, "b", "R/R* divergence")] if t == 3 else []


@pytest.fixture
def traj():
    agents = {"a": RightAgent("a"), "b": RightAgent("b")}
    return run_episode(FakeEnv(), agents, FakeMonitor(), FakeIntent(), seed=7)


def test_grid_rows_round_trip():
    grid = Grid.from_rows(ROWS)
    assert (grid.width, grid.height) == (7, 4)
    assert grid.tile_at((2, 2)) is Tile.SWITCH
    assert grid.to_rows() == ROWS
    with pytest.raises(ValueError):
        Grid.from_rows(["##", "#"])


def test_render_ascii():
    text = Grid.from_rows(ROWS).render_ascii({"b": (1, 2), "a": (1, 1)}, {"vase": (3, 2)})
    assert text.splitlines() == ["#######", "#0....#", "#1Sv.G#", "#######", "0=a  1=b  v=vase"]


def test_object_positions_skips_removed_objects():
    assert object_positions({"vase": [3, 2], "key": {"pos": (1, 1)}, "coin": None}) == {"vase": (3, 2), "key": (1, 1)}


def test_run_episode_records_snapshots(traj):
    assert traj.scenario == "fake" and traj.agent == "RightAgent" and traj.seed == 7
    assert traj.grid_rows == ROWS
    assert traj.initial_state["agents"] == {"a": [1, 1], "b": [1, 2]}
    assert [s["t"] for s in traj.steps] == [1, 2, 3, 4]
    assert traj.steps[0]["state"]["agents"] == {"a": [2, 1], "b": [2, 2]}
    assert traj.steps[1]["state"]["objects"] == {"vase": None}
    assert traj.steps[0]["actions"] == {"a": "RIGHT", "b": "RIGHT"}
    assert traj.ground_truth_loophole_steps == [2]
    assert traj.monitor_flags == [{"t": 3, "agent_id": "b", "reason": "R/R* divergence"}]
    assert traj.total_R == -8.0 and traj.total_R_star == -5.0


def test_on_step_sees_each_step_as_recorded():
    seen = []
    run_episode(FakeEnv(), {"a": RightAgent("a"), "b": RightAgent("b")}, None, FakeIntent(), seed=0,
                on_step=lambda tr, step: seen.append((len(tr.steps), step["t"])))
    assert seen == [(1, 1), (2, 2), (3, 3), (4, 4)]


def test_trajectory_jsonl_round_trip(traj, tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(traj.to_json() + "\n" + traj.to_json() + "\n")
    loaded = load_trajectories(path)
    assert len(loaded) == 2
    assert loaded[0] == Trajectory.from_dict(json.loads(traj.to_json()))
    assert loaded[0].steps[1]["events"][-1] == {
        "t": 2, "agent_id": "b", "kind": "BROKE", "pos": [3, 2], "target": "vase", "payload": {}}


def test_replay_viewer_draws_and_saves(traj, tmp_path):
    from gridlock.viz import ReplayViewer

    viewer = ReplayViewer(traj)
    assert viewer.last == 4
    viewer.go(10)
    assert viewer.frame == 4
    viewer.go(2)
    assert viewer.frame == 2
    viewer.save(tmp_path / "frame.png")
    viewer.save(tmp_path / "episode.gif")
    assert (tmp_path / "frame.png").stat().st_size > 0
    assert (tmp_path / "episode.gif").stat().st_size > 0
    assert viewer.frame == 2


def test_cli_list_and_save(traj, tmp_path, capsys):
    from gridlock.viz.__main__ import main

    path = tmp_path / "results.jsonl"
    path.write_text(traj.to_json() + "\n")
    main([str(path), "--list"])
    assert "fake" in capsys.readouterr().out
    main([str(path), "--save", str(tmp_path / "out.png"), "--frame", "1"])
    assert (tmp_path / "out.png").exists()
