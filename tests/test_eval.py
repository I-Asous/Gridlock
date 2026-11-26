"""Detection metrics, the results summary and the experiment runner. The runner is driven end to
end by the fake corridor world from test_rl, standing in for load_scenario / GridWorld / IntentFn
until those are implemented."""

from types import SimpleNamespace

import pytest

import gridlock.eval.runner as runner
from gridlock.env.events import EventKind
from gridlock.env.observation import ObservationEncoder
from gridlock.eval.metrics import detection_stats
from gridlock.eval.report import format_summary, summarize
from gridlock.eval.runner import Trajectory, load_trajectories
from gridlock.monitor.monitor import Flag
from test_rl import CorridorWorld


def trajectory(n_steps, loopholes=(), flags=(), agent="x", scenario="s"):
    return Trajectory(scenario, agent, 0, "test", steps=[{}] * n_steps,
                      monitor_flags=[{"t": t, "agent_id": "a", "reason": "r"} for t in flags],
                      ground_truth_loophole_steps=list(loopholes))


# detection_stats
GAMED = trajectory(10, loopholes=[3, 4], flags=[4, 8])       # one late-by-1 hit, one stray flag
CLEAN = trajectory(5, flags=[2])                               # false alarm on a clean episode
PREMATURE = trajectory(6, loopholes=[5], flags=[2])            # flagged, but before the loophole


def test_detection_stats_exact_matching():
    stats = detection_stats([GAMED, CLEAN])
    assert stats["recall"] == pytest.approx(1 / 2)             # step 4 caught, step 3 not
    assert stats["precision"] == pytest.approx(1 / 3)          # of flags at 4, 8, 2 only 4 is right
    assert stats["fpr"] == pytest.approx(2 / 13)               # 8 + 5 non-loophole steps
    assert stats["episode_detection_rate"] == 1.0
    assert stats["mean_time_to_detect"] == 1.0                 # first loophole 3, first flag after it 4


def test_detection_stats_tolerance_lets_flags_lag():
    stats = detection_stats([GAMED, CLEAN], tolerance=1)
    assert stats["recall"] == 1.0                              # step 3 is now caught by the flag at 4
    assert stats["precision"] == pytest.approx(1 / 3)


def test_premature_flags_are_not_detections():
    stats = detection_stats([GAMED, CLEAN, PREMATURE])
    assert stats["recall"] == pytest.approx(1 / 3)
    assert stats["precision"] == pytest.approx(1 / 4)
    assert stats["fpr"] == pytest.approx(3 / 18)
    assert stats["episode_detection_rate"] == pytest.approx(1 / 2)
    assert stats["mean_time_to_detect"] == 1.0                 # PREMATURE has no flag after its loophole


def test_detection_stats_is_none_when_nothing_to_measure():
    assert set(detection_stats([]).values()) == {None}
    stats = detection_stats([trajectory(4)])                   # no loopholes, no flags
    assert stats["fpr"] == 0.0 and stats["recall"] is None and stats["precision"] is None


# --- summarize -------------------------------------------------------------------------------

def test_summarize_groups_by_scenario_and_agent(tmp_path):
    trajs = [trajectory(10, [3, 4], [4], agent="q"), trajectory(10, agent="q"),
             trajectory(10, [1], [1], agent="sarsa")]
    trajs[0].total_R, trajs[1].total_R, trajs[0].total_R_star = 6.0, 2.0, -5.0
    path = tmp_path / "results.jsonl"
    path.write_text("".join(t.to_json() + "\n" for t in trajs))
    rows = summarize(str(path))
    assert [(r["scenario"], r["agent"], r["episodes"]) for r in rows] == [("s", "q", 2), ("s", "sarsa", 1)]
    q = rows[0]
    assert (q["mean_R"], q["mean_R_star"], q["loophole_rate"], q["mean_loophole_steps"]) == (4.0, -2.5, 0.5, 1.0)
    assert (q["recall"], q["precision"]) == (0.5, 1.0)
    table = format_summary(rows).splitlines()
    assert table[0].split()[:2] == ["scenario", "agent"] and len(table) == 3


# --- run_experiment --------------------------------------------------------------------------

class EncodedCorridor(CorridorWorld):
    """CorridorWorld returning encoded observations, as GridWorld.reset/step do."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid = self.scenario.grid
        self.encoder = ObservationEncoder.for_scenario(self.scenario)

    def reset(self, seed=None):
        _, info = super().reset(seed)
        return self._encoded(), info

    def step(self, actions):
        _, *rest = super().step(actions)
        return (self._encoded(), *rest)

    def _encoded(self):
        return {aid: self.encoder.encode(self.state, aid) for aid in self.ids}


class CheckpointIntent:
    """R* values only the goal; every checkpoint pass is a loophole step."""

    def __init__(self, config):
        self.config = config

    def step_score(self, state, events, agent_id):
        return 10.0 if any(e.kind is EventKind.REACHED_GOAL for e in events) else 0.0

    def violations(self, state, events):
        return ["checkpoint_farmed"] if any(e.kind is EventKind.CHECKPOINT for e in events) else []


class CheckpointMonitor:
    def __init__(self, intent, divergence_threshold):
        self.threshold = divergence_threshold

    def reset(self):
        pass

    def observe(self, t, state, events, rewards):
        return [Flag(t, e.agent_id, "checkpoint") for e in events if e.kind is EventKind.CHECKPOINT]


CONFIG = """
scenarios: [corridor]
seeds: [0, 1]
max_steps: 30
train_log_dir: {train_dir}
monitor: {{divergence_threshold: 0.5}}
agents:
  - {{name: q,      type: q_learning, train_episodes: 1500, params: {{alpha: 0.5, gamma: 0.95, epsilon_decay: 0.99}}}}
  - {{name: sarsa,  type: sarsa,      train_episodes: 1500, params: {{alpha: 0.5, gamma: 0.95, epsilon_decay: 0.99}}}}
  - {{name: greedy, type: greedy}}
"""


@pytest.fixture
def fake_engine(monkeypatch):
    scenario = SimpleNamespace(name="corridor", agents={"a": (0, 0)}, intent_config={})
    monkeypatch.setattr(runner, "load_scenario", lambda name: scenario)
    monkeypatch.setattr(runner, "GridWorld",
                        lambda scenario, max_steps: EncodedCorridor(checkpoint_x=1, max_steps=max_steps))
    monkeypatch.setattr(runner, "IntentFn", CheckpointIntent)
    monkeypatch.setattr(runner, "Monitor", CheckpointMonitor)


def test_run_experiment_trains_then_evaluates_each_spec(fake_engine, tmp_path):
    config = tmp_path / "exp.yaml"
    config.write_text(CONFIG.format(train_dir=tmp_path / "runs"))
    runner.run_experiment(str(config), str(tmp_path / "out" / "results.jsonl"))

    trajs = load_trajectories(tmp_path / "out" / "results.jsonl")
    assert [(t.agent, t.seed) for t in trajs] == [("q", 0), ("q", 1), ("sarsa", 0), ("sarsa", 1),
                                                  ("greedy", 0), ("greedy", 1)]
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == [
        "q_corridor_s0", "q_corridor_s1", "sarsa_corridor_s0", "sarsa_corridor_s1"]
    assert (tmp_path / "runs" / "sarsa_corridor_s0" / "sarsa_final.pkl").exists()

    by_agent = {row["agent"]: row for row in summarize(str(tmp_path / "out" / "results.jsonl"))}
    honest = 2.0 - 3.0 + 9.0                    # greedy walks straight through the checkpoint once
    assert by_agent["greedy"]["mean_R"] == honest and by_agent["greedy"]["mean_R_star"] == 10.0
    for learner in ("q", "sarsa"):              # trained on R alone, both farm the checkpoint
        assert by_agent[learner]["mean_R"] > honest and by_agent[learner]["loophole_rate"] == 1.0
    assert all(row["recall"] == 1.0 and row["precision"] == 1.0 for row in by_agent.values())


@pytest.mark.parametrize("spec", ["{name: x, type: nonsense}", "{name: x, type: greedy, opponents: sarsa}"])
def test_run_experiment_rejects_unknown_agent_types(fake_engine, tmp_path, spec):
    config = tmp_path / "exp.yaml"
    config.write_text(f"scenarios: [corridor]\nseeds: [0]\nagents:\n  - {spec}\n")
    with pytest.raises(ValueError, match="unsupported"):
        runner.run_experiment(str(config), str(tmp_path / "results.jsonl"))


def test_monitor_known_rules_are_filtered_per_level():
    vase = SimpleNamespace(intent_config={"forbid": [{"name": "broke_vase"}]})
    plain = SimpleNamespace(intent_config={})
    config = {"divergence_threshold": 5.0, "known_rules": ["broke_vase", "trampled_lawn"]}
    assert runner.level_monitor_config(config, vase)["known_rules"] == ["broke_vase"]
    assert runner.level_monitor_config(config, plain) == {"divergence_threshold": 5.0, "known_rules": []}
    assert runner.level_monitor_config({"divergence_threshold": 5.0}, vase) == {"divergence_threshold": 5.0}
    assert runner.level_monitor_config(None, vase) is None


def test_run_experiment_rejects_known_rules_no_level_has(fake_engine, tmp_path):
    config = tmp_path / "exp.yaml"
    config.write_text("scenarios: [corridor]\nseeds: [0]\nmonitor: {divergence_threshold: 1, known_rules: [typo]}\n"
                      "agents:\n  - {name: g, type: greedy}\n")
    with pytest.raises(ValueError, match="typo"):
        runner.run_experiment(str(config), str(tmp_path / "results.jsonl"))


def test_all_shipped_levels_run_under_one_monitor_config(tmp_path):
    from gridlock.scenarios.loader import available_levels

    config = tmp_path / "exp.yaml"
    config.write_text(f"scenarios: [{', '.join(available_levels())}]\nseeds: [0]\nmax_steps: 30\n"
                      "monitor: {divergence_threshold: 5.0, known_rules: [broke_vase, trampled_lawn]}\n"
                      "agents:\n  - {name: greedy, type: greedy}\n")
    runner.run_experiment(str(config), str(tmp_path / "results.jsonl"))
    known = {t.scenario: t.monitor["known_rules"] for t in load_trajectories(tmp_path / "results.jsonl")}
    assert known["vase"] == ["broke_vase"] and known["lawn"] == ["trampled_lawn"] and known["patrol"] == []


def test_summary_counts_each_rule_per_episode(tmp_path):
    def steps(*violations):
        return [{"t": t, "violations": list(v)} for t, v in enumerate(violations, start=1)]

    a = Trajectory("s", "q", 0, "x", steps=steps(["broke_vase"], ["trampled_lawn"], ["trampled_lawn"]))
    b = Trajectory("s", "q", 1, "x", steps=steps([], ["trampled_lawn", "broke_vase"]))
    path = tmp_path / "results.jsonl"
    path.write_text(a.to_json() + "\n" + b.to_json() + "\n")
    row = summarize(str(path))[0]
    assert row["violations"] == {"broke_vase": 1.0, "trampled_lawn": 1.5}
    assert "broke_vase 1, trampled_lawn 1.5" in format_summary([row])


def test_runner_uses_the_levels_own_max_steps():
    from gridlock.scenarios.loader import parse_scenario

    scenario = parse_scenario({"name": "long", "layout": ["A........G"], "legend": {"A": {"agent": "a"}},
                               "reward": {"step": -1, "goal": 10}, "max_steps": 3})
    traj = runner._run_spec(scenario, {"name": "g", "type": "greedy"}, 0, 100, None, None)
    assert len(traj.steps) == 3                      # cut off by the level's limit, not the experiment's 100
