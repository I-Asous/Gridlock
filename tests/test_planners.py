"""The A* and CSP planners and the small CSP solver behind the CSP planner."""

from dataclasses import replace

import pytest

from gridlock.agents import astar as astar_module
from gridlock.agents.astar import AStarAgent
from gridlock.agents.csp import CSPAgent, Problem, SearchBudgetExceeded, Solver
from gridlock.env.events import EventKind
from gridlock.env.grid import Action
from gridlock.env.gridworld import GridWorld
from gridlock.eval.levelcheck import check_level
from gridlock.eval.runner import load_trajectories, run_experiment
from gridlock.scenarios.loader import available_levels, load_scenario
from gridlock.scenarios.loader import parse_scenario


def world(layout, legend=None, reward=None, max_steps=100, **extra):
    spec = {"layout": layout, "legend": legend or {"A": {"agent": "a"}},
            "reward": reward or {"step": -1, "goal": 10}, **extra}
    w = GridWorld(parse_scenario(spec), max_steps=max_steps)
    w.reset()
    return w


def level(name, max_steps=100):
    w = GridWorld(load_scenario(name), max_steps=max_steps)
    w.reset()
    return w


def play(w, agent_id, plan):
    """Simulate a plan from the current state: (total R, final state, every event)."""
    state, total, events = w.get_state(), 0.0, []
    for action in plan:
        state, rewards, step_events = w.simulate(state, {agent_id: action})
        total += rewards[agent_id]
        events += step_events
    return total, state, events


def event_kinds(events):
    return {event.kind for event in events}


# --- A* --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", available_levels())
def test_astar_finds_the_r_optimal_play_on_every_level(name):
    w = level(name)
    agent = AStarAgent(sorted(w.scenario.agents)[0])
    agent.reset(w)
    total, _, _ = play(w, agent.agent_id, agent.plan)
    assert total == pytest.approx(check_level(w.scenario, max_steps=100).best.R)


def test_astar_takes_the_vase_loophole():
    w = level("vase")
    agent = AStarAgent("a0")
    agent.reset(w)
    total, state, events = play(w, "a0", agent.plan)
    assert len(agent.plan) == 8 and total == 2.0
    assert "a0" in state.done and EventKind.BROKE in event_kinds(events)


def test_astar_farms_bonus_points_when_that_pays_more():
    w = level("midfield")
    agent = AStarAgent("a0")
    agent.reset(w)
    _, _, events = play(w, "a0", agent.plan)
    repeats = [e for e in events if e.kind is EventKind.CHECKPOINT and not e.payload["first"]]
    assert len(repeats) > 10


def test_astar_stays_exact_when_the_reward_bound_is_too_low(monkeypatch):
    """With M below the real best step reward, costs would go negative; the search must notice,
    raise M and still return the optimum."""
    monkeypatch.setattr(astar_module, "_reward_bounds", lambda reward_fn: (-5.0, 0.0))
    w = level("midfield")
    agent = AStarAgent("a0")
    agent.reset(w)
    total, _, _ = play(w, "a0", agent.plan)
    assert total == pytest.approx(check_level(w.scenario).best.R)


def test_astar_respects_the_step_limit_and_unreachable_goals():
    for w in (world(["A....G"], max_steps=3), world(["A.#G"], max_steps=12)):   # goal too far / walled off
        agent = AStarAgent("a")
        agent.reset(w)
        _, state, _ = play(w, "a", agent.plan)
        assert len(agent.plan) == w.max_steps and "a" not in state.done


def test_astar_budget_returns_a_partial_plan():
    w = level("midfield")
    agent = AStarAgent("a0", max_expansions=50)
    agent.reset(w)
    assert agent.expansions == 51 and 0 < len(agent.plan) <= 100


@pytest.mark.parametrize("cls", [AStarAgent, CSPAgent])
def test_planners_replan_when_the_world_diverges(cls):
    w = world(["A....", ".....", "....G"])
    agent = cls("a")
    agent.reset(w)
    first = agent.act(None)
    w.step({"a": first})
    # Something outside the plan moves the agent back to the start.
    w.set_state(replace(w.get_state(), agents=(("a", (0, 0)),)))
    agent.act(None)
    assert agent.replans == 1
    for _ in range(10):
        if "a" in w.get_state().done:
            break
        w.step({"a": agent.act(None)})
    assert "a" in w.get_state().done


# --- CSP -------------------------------------------------------------------------------------------

def test_csp_takes_the_shortest_route_even_through_a_loophole():
    w = level("vase")
    agent = CSPAgent("a0")
    agent.reset(w)
    total, state, events = play(w, "a0", agent.plan)
    assert len(agent.plan) == 8 and "a0" in state.done and EventKind.BROKE in event_kinds(events)


@pytest.mark.parametrize("constraint", [
    {"name": "broke_vase", "kind": "BROKE", "target": "vase", "penalty": -20},     # an intent rule, as is
    lambda state, action, next_state, events: not any(e.kind is EventKind.BROKE for e in events),
])
def test_csp_extra_constraints_rule_out_the_loophole(constraint):
    w = level("vase")
    agent = CSPAgent("a0", extra_constraints=[constraint])
    agent.reset(w)
    total, state, events = play(w, "a0", agent.plan)
    assert len(agent.plan) == 19 and total == -9.0
    assert "a0" in state.done and EventKind.BROKE not in event_kinds(events)


def test_csp_goes_straight_for_the_goal_and_never_farms():
    w = level("midfield")
    agent = CSPAgent("a0")
    agent.reset(w)
    _, state, events = play(w, "a0", agent.plan)
    assert len(agent.plan) == 8 and "a0" in state.done
    assert not event_kinds(events) & {EventKind.CHECKPOINT, EventKind.TRAPPED, EventKind.BROKE}


def test_csp_routes_around_traps():
    w = world(["A^G", "..."])
    agent = CSPAgent("a")
    agent.reset(w)
    _, state, events = play(w, "a", agent.plan)
    assert len(agent.plan) == 4 and "a" in state.done and EventKind.TRAPPED not in event_kinds(events)


def test_csp_stays_put_without_a_plan():
    agent = CSPAgent("a0", horizon=10, extra_constraints=[{"kind": "BROKE"}])   # the clean route needs 19
    agent.reset(level("vase"))
    assert agent.plan == [] and agent.act(None) is Action.STAY
    agent = CSPAgent("a")
    agent.reset(world(["A#G"]))
    assert agent.plan == []


def test_csp_rejects_malformed_constraints():
    with pytest.raises(ValueError):
        CSPAgent("a", extra_constraints=[{"target": "vase"}])


# --- the solver ------------------------------------------------------------------------------------

def coloring(colors):
    p = Problem()
    regions = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]
    for region in regions:
        p.add_variable(region, colors)
    borders = [("WA", "NT"), ("WA", "SA"), ("NT", "SA"), ("NT", "Q"), ("SA", "Q"), ("SA", "NSW"), ("SA", "V"),
               ("Q", "NSW"), ("NSW", "V")]
    for a, b in borders:
        p.add_binary(a, b, lambda x, y: x != y)
    return p, borders


def test_solver_colours_a_map_and_proves_two_colours_impossible():
    p, borders = coloring(["red", "green", "blue"])
    solution = Solver(p).solve()
    assert solution is not None and all(solution[a] != solution[b] for a, b in borders)
    assert Solver(coloring(["red", "green"])[0]).solve() is None


def test_solver_handles_nary_constraints_with_and_without_a_tuple_lister():
    for lister in (None, lambda xs, ys, zs: ((x, y, z) for x in xs for y in ys for z in zs if x + y == z)):
        p = Problem()
        for var in "xyz":
            p.add_variable(var, range(6))
        p.add_nary(("x", "y", "z"), lambda x, y, z: x + y == z, lister)
        p.add_binary("x", "y", lambda x, y: x > y)
        p.add_variable("w", [5])
        p.add_binary("w", "z", lambda w, z: w == z)
        solution = Solver(p).solve()
        assert solution["x"] + solution["y"] == solution["z"] == 5 and solution["x"] > solution["y"]


def test_solver_budget():
    p = Problem()
    for i in range(8):
        p.add_variable(i, range(8))
    for i in range(8):
        for j in range(i + 1, 8):
            p.add_binary(i, j, lambda a, b, d=j - i: a != b and abs(a - b) != d)
    with pytest.raises(SearchBudgetExceeded):
        Solver(p, max_nodes=2).solve()
    queens = Solver(p).solve()                        # eight queens
    assert len(set(queens.values())) == 8


# --- through the experiment runner -----------------------------------------------------------------

def test_runner_runs_planners_from_a_config(tmp_path):
    config = tmp_path / "planners.yaml"
    config.write_text(
        "scenarios: [vase]\nseeds: [0]\nagents:\n"
        "  - {name: astar, type: astar}\n"
        "  - {name: csp, type: csp}\n"
        "  - {name: csp_intent, type: csp, params: {extra_constraints: [{kind: BROKE, target: vase}]}}\n")
    run_experiment(str(config), str(tmp_path / "out.jsonl"))
    trajs = {tr.agent: tr for tr in load_trajectories(tmp_path / "out.jsonl")}
    assert trajs["astar"].ground_truth_loophole_steps and trajs["csp"].ground_truth_loophole_steps
    assert trajs["csp_intent"].ground_truth_loophole_steps == []
    assert all("a0" in tr.steps[-1]["state"]["done"] for tr in trajs.values())
