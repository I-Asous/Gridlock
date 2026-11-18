"""Rendering, trajectory recording and the replay viewer, driven by a tiny fake environment."""

import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest

from gridlock.agents.base import Agent
from gridlock.agents.scripted import GreedyAgent
from gridlock.env.events import Event, EventKind
from gridlock.env.grid import Action, Grid, Tile
from gridlock.env.gridworld import State, object_positions
from gridlock.env.observation import ObservationEncoder
from gridlock.eval.runner import Trajectory, load_trajectories, run_episode
from gridlock.monitor.monitor import Flag

ROWS = [
    "##########",
    "#........#",
    "#.S......#",
    "#........#",
    "#....C...#",
    "#........#",
    "#........#",
    "#........#",
    "#.......G#",
    "##########",
]


class FakeEnv:
    """Agents a (1, 1) and b (1, 2) move on ROWS; walls block. Walking onto an intact object
    breaks it. R: -1 per step, +10 on reaching G, after which that agent is done. The run is cut
    off after `max_steps`. The "sensor_off" flag is set from t=3 so the viewer has one to show."""

    def __init__(self, max_steps=4):
        self.grid = Grid.from_rows(ROWS)
        self.scenario = SimpleNamespace(name="fake", grid=self.grid, agents={"a": (1, 1), "b": (1, 2)})
        self.encoder = ObservationEncoder(self.grid.width, self.grid.height, ["a", "b"], ["vase"])
        self.max_steps = max_steps
        self.state = None

    def reset(self, seed=None):
        self.state = State(t=0, agents=(("a", (1, 1)), ("b", (1, 2))), objects=(("vase", (3, 2)),))
        return self._observations(), {}

    def get_state(self):
        return self.state

    def step(self, actions):
        t = self.state.t + 1
        objects = dict(self.state.objects)
        agents, events, rewards, at_goal = [], [], {}, {}
        for aid, (x, y) in self.state.agents:
            already_home = self.grid.tile_at((x, y)) is Tile.GOAL
            dx, dy = actions[aid].value or (0, 0)
            if (dx, dy) != (0, 0) and self.grid.is_walkable((x + dx, y + dy)):
                x, y = x + dx, y + dy
                events.append(Event(t, aid, EventKind.MOVED, (x, y)))
                for oid, pos in objects.items():
                    if pos is not None and tuple(pos) == (x, y):
                        objects[oid] = None
                        events.append(Event(t, aid, EventKind.BROKE, (x, y), target=oid))
            at_goal[aid] = self.grid.tile_at((x, y)) is Tile.GOAL
            if at_goal[aid] and not already_home:
                events.append(Event(t, aid, EventKind.REACHED_GOAL, (x, y)))
            rewards[aid] = 0.0 if already_home else (9.0 if at_goal[aid] else -1.0)
            agents.append((aid, (x, y)))
        flags = frozenset({"sensor_off"}) if t >= 3 else frozenset()
        self.state = State(t, tuple(agents), tuple(objects.items()), flags)
        truncated = {aid: t >= self.max_steps and not done for aid, done in at_goal.items()}
        return self._observations(), rewards, at_goal, truncated, {"events": events}

    def _observations(self):
        return {aid: self.encoder.encode(self.state, aid) for aid, _ in self.state.agents}


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
    assert (grid.width, grid.height) == (10, 10)
    assert grid.tile_at((2, 2)) is Tile.SWITCH
    assert grid.to_rows() == ROWS
    with pytest.raises(ValueError):
        Grid.from_rows(["##", "#"])


def test_render_ascii():
    text = Grid.from_rows(ROWS).render_ascii({"b": (1, 2), "a": (1, 1)}, {"vase": (3, 2)})
    lines = text.splitlines()
    assert len(lines) == 11
    assert lines[1:3] == ["#0.......#", "#1Sv.....#"]
    assert lines[-1] == "0=a  1=b  v=vase"


def test_distances_route_around_walls():
    grid = Grid.from_rows(["#####", "#.#G#", "#...#", "#####"])
    distance = grid.distances_from(grid.cells(Tile.GOAL))
    assert grid.cells(Tile.GOAL) == [(3, 1)]
    assert distance[(3, 1)] == 0 and distance[(1, 1)] == 4    # down, right, right, up
    assert (2, 1) not in distance                            # walls are never reachable


def test_greedy_agent_steps_toward_goal():
    grid = Grid.from_rows(["#####", "#.#G#", "#...#", "#####"])
    agent = GreedyAgent("a")
    agent.reset(SimpleNamespace(grid=grid))
    assert agent.act([1, 1]) is Action.DOWN       # right is a wall
    assert agent.act([3, 2]) is Action.UP
    assert agent.act([3, 1]) is Action.STAY       # on the goal


def test_greedy_agents_reach_goal():
    agents = {"a": GreedyAgent("a"), "b": GreedyAgent("b")}
    traj = run_episode(FakeEnv(max_steps=30), agents, FakeMonitor(), FakeIntent(), seed=7)
    assert traj.steps[-1]["state"]["agents"] == {"a": [8, 8], "b": [8, 8]}
    reached = [(e["agent_id"], s["t"]) for s in traj.steps for e in s["events"] if e["kind"] == "REACHED_GOAL"]
    assert reached == [("b", 13), ("a", 14)] and len(traj.steps) == 14
    assert traj.ground_truth_loophole_steps == [2]           # b's shortest path runs through the vase
    assert traj.total_R == (-13 + 9) + (-12 + 9)


def test_episode_ends_when_every_agent_is_done_or_cut_off():
    agents = {"a": GreedyAgent("a"), "b": GreedyAgent("b")}
    traj = run_episode(FakeEnv(max_steps=13), agents, None, FakeIntent(), seed=0)
    assert len(traj.steps) == 13                              # b home at t=13, a out of time
    assert traj.steps[-1]["state"]["agents"] == {"a": [8, 7], "b": [8, 8]}


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
