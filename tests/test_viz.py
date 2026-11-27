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


def _relabel(traj, agent, seed=7, scenario="fake"):
    copy = Trajectory.from_dict(json.loads(traj.to_json()))
    copy.agent, copy.seed, copy.scenario = agent, seed, scenario
    return copy


def test_pick_comparison_takes_one_episode_per_agent(traj):
    from gridlock.viz.__main__ import pick_comparison

    trajs = [_relabel(traj, "q", 0), _relabel(traj, "q", 1), _relabel(traj, "sarsa", 0),
             _relabel(traj, "q", 0), _relabel(traj, "greedy", 0, scenario="other")]
    assert [(t.agent, t.seed) for t in pick_comparison(trajs)] == [("q", 0), ("sarsa", 0)]
    assert [(t.agent, t.seed) for t in pick_comparison(trajs, seed=1)] == [("q", 1)]
    assert [t.agent for t in pick_comparison(trajs, scenario="other")] == ["greedy"]
    with pytest.raises(ValueError, match="scenarios in the file"):
        pick_comparison(trajs, scenario="missing")
    with pytest.raises(ValueError, match="seeds"):
        pick_comparison(trajs, seed=9)


def test_comparison_viewer_steps_every_agent_together(traj, tmp_path):
    from gridlock.viz import ComparisonViewer

    short = _relabel(traj, "short")
    short.steps = short.steps[:2]                                    # ends before the others
    viewer = ComparisonViewer([_relabel(traj, "q"), short])
    dash = viewer.dash
    assert viewer.last == 4 and dash.studied == ["a", "a"]
    viewer.go(3)
    assert [dash._status(k, 3) for k in range(2)] == ["t = 3", "cut off at t = 2"]
    pills = [text.get_text() for text in dash.status_texts]
    assert pills == ["RUNNING  t = 3", "CUT OFF  t = 2"]
    assert dash.score_texts[0].get_text().startswith("r ")
    assert dash._log_lines(1, 3)[0][0] == "episode over"
    viewer.go(0)
    assert [text.get_text() for text in dash.status_texts] == ["START", "START"]
    assert dash._pill("reached the goal at t = 8") == ("⚑ GOAL  t = 8", dash._pill("reached the goal at t = 1")[1])
    viewer.save(tmp_path / "cmp.png")
    viewer.save(tmp_path / "cmp.gif")
    assert (tmp_path / "cmp.png").stat().st_size > 0 and (tmp_path / "cmp.gif").stat().st_size > 0


def test_cli_compares_all_agents_by_default(traj, tmp_path, capsys):
    from gridlock.viz.__main__ import main

    path = tmp_path / "results.jsonl"
    path.write_text("".join(_relabel(traj, name).to_json() + "\n" for name in ("q", "sarsa", "greedy")))
    main([str(path), "--save", str(tmp_path / "all.png")])
    assert "fake, seed 7: q, sarsa, greedy" in capsys.readouterr().out
    main([str(path), "-e", "1", "--compare", "--save", str(tmp_path / "all2.png")])
    main([str(path), "-e", "1", "--save", str(tmp_path / "one.png")])          # single-episode view still works
    assert all((tmp_path / name).exists() for name in ("all.png", "all2.png", "one.png"))


def test_viewers_fit_a_laptop_screen_and_refit_on_resize(traj):
    from gridlock.viz import ComparisonViewer, ReplayViewer

    for viewer in (ReplayViewer(traj), ComparisonViewer([_relabel(traj, "q"), _relabel(traj, "sarsa")])):
        assert tuple(viewer.fig.get_size_inches()) == (14.0, 8.0)
        viewer.go(2)
        viewer.fig.set_size_inches(11, 6.5)
        viewer._on_resize(None)                                   # redraws at the new size, same frame
        assert viewer.frame == 2 and viewer._size == (11.0, 6.5)


def _results(traj, tmp_path):
    trajs = [_relabel(traj, "q", 0), _relabel(traj, "sarsa", 0), _relabel(traj, "q", 1),
             _relabel(traj, "q", 0, scenario="other")]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(tr.to_json() + "\n" for tr in trajs))
    return trajs, path


def test_level_entries_cover_the_file_then_unrecorded_shipped_levels(traj):
    from gridlock.scenarios.loader import available_levels
    from gridlock.viz.selection import level_entries

    trajs = [_relabel(traj, "q", 0), _relabel(traj, "sarsa", 0), _relabel(traj, "q", 1),
             _relabel(traj, "q", 0, scenario="vase")]
    entries = level_entries(trajs)
    assert [e["name"] for e in entries][:2] == ["fake", "vase"]
    fake = entries[0]
    assert fake["recorded"] and fake["agent_names"] == ["q", "sarsa"] and fake["seeds"] == [0, 1]
    assert fake["episodes"] == 3 and fake["loophole_episodes"] == 3 and fake["flagged_episodes"] == 3
    shipped = {e["name"]: e for e in entries[2:]}
    assert set(shipped) == set(available_levels()) - {"vase"}
    assert all(not e["recorded"] and e["grid_rows"] for e in shipped.values())
    assert [e["name"] for e in level_entries(trajs, include_shipped=False)] == ["fake", "vase"]


def test_picker_opens_a_level_and_comes_back(traj, tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent, MouseEvent
    from gridlock.viz.picker import LevelPicker

    trajs, _ = _results(traj, tmp_path)
    picker = LevelPicker(trajs, source="results.jsonl", include_shipped=False)
    assert [card["name"] for card in picker.cards] == ["fake", "other"]
    assert [chip.seed for chip in picker.cards[0]["chips"]] == [0, 1]

    picker.fig.canvas.draw()                                       # lay out the chips before clicking them
    chip = picker.cards[0]["chips"][1]
    box = chip.get_window_extent()
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    picker.fig.canvas.callbacks.process("button_press_event",
                                        MouseEvent("button_press_event", picker.fig.canvas, cx, cy, button=1))
    viewer = picker.viewer
    assert {(tr.scenario, tr.seed, tr.agent) for tr in viewer.dash.trajs} == {("fake", 1, "q")}
    assert not plt.fignum_exists(picker.fig.number)                 # the picker window closed

    viewer.fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", viewer.fig.canvas, "escape"))
    assert picker.viewer is None and plt.fignum_exists(picker.fig.number)
    assert not plt.fignum_exists(viewer.fig.number)

    picker.fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", picker.fig.canvas, "2"))
    assert [tr.agent for tr in picker.viewer.dash.trajs] == ["q"] and picker.viewer.dash.trajs[0].scenario == "other"
    plt.close("all")


def test_picker_ignores_unrecorded_levels(traj):
    from gridlock.viz.picker import LevelPicker

    picker = LevelPicker([_relabel(traj, "q")])
    unrecorded = [k for k, card in enumerate(picker.cards) if not card["recorded"]]
    assert unrecorded and all(not picker.cards[k]["chips"] for k in unrecorded)
    picker._on_key(SimpleNamespace(key=str(unrecorded[0] + 1)))
    assert picker.viewer is None


def test_cli_opens_the_level_picker_by_default(traj, tmp_path, monkeypatch):
    from gridlock.viz import __main__ as cli
    from gridlock.viz.picker import LevelPicker

    _, path = _results(traj, tmp_path)
    shown = []
    monkeypatch.setattr(LevelPicker, "show", lambda self: shown.append([c["name"] for c in self.cards]))
    cli.main([str(path)])
    assert shown and shown[0][:2] == ["fake", "other"]
    monkeypatch.setattr("gridlock.viz.replay.ReplayViewer.show", lambda self: shown.append("compare"))
    cli.main([str(path), "--no-picker"])
    cli.main([str(path), "--scenario", "other"])
    assert shown[1:] == ["compare", "compare"]


def test_big_boards_draw_as_one_image():
    import matplotlib.pyplot as plt

    rows = ["#" * 50] + ["#" + "." * 48 + "#" for _ in range(48)] + ["#" * 50]
    rows[1] = "#^" + "." * 46 + "G#"
    fig, ax = plt.subplots()
    Grid.from_rows(rows).draw_board(ax)
    assert len(ax.images) == 1
    assert len(ax.patches) == 2                      # the trap's hatching and the outer frame, not 2,500 cells
    small, ax_small = plt.subplots()
    Grid.from_rows(ROWS).draw_board(ax_small)
    assert not ax_small.images and len(ax_small.patches) > 100
    plt.close("all")
