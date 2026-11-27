"""Choosing what to view from a results file: which levels it holds and which episodes to compare.

No matplotlib here, so the CLI can use it before it picks a backend.
"""

from gridlock.env.gridworld import object_positions


def pick_comparison(trajectories, scenario=None, seed=None) -> list:
    """One trajectory per agent name (the first in the file) for one scenario and seed. Missing
    scenario or seed: the first one in the file."""
    if scenario is None:
        scenario = trajectories[0].scenario
    on_level = [tr for tr in trajectories if tr.scenario == scenario]
    if not on_level:
        raise ValueError(f"no episodes for scenario {scenario!r}; scenarios in the file: "
                         f"{', '.join(dict.fromkeys(tr.scenario for tr in trajectories))}")
    if seed is None:
        seed = on_level[0].seed
    picked = {}
    for tr in on_level:
        if tr.seed == seed:
            picked.setdefault(tr.agent, tr)
    if not picked:
        raise ValueError(f"no episodes for scenario {scenario!r} with seed {seed}; seeds: "
                         f"{', '.join(str(s) for s in dict.fromkeys(tr.seed for tr in on_level))}")
    return list(picked.values())


def level_entries(trajectories, include_shipped: bool = True) -> list[dict]:
    """One entry per level: the levels in the file (in file order), then the shipped levels that
    have no episodes in it. Each entry has name, recorded, grid_rows, agents (start positions),
    objects, description, and for recorded levels agent_names, seeds, episodes, loophole_episodes
    and flagged_episodes."""
    entries = {}
    for tr in trajectories:
        entry = entries.get(tr.scenario)
        if entry is None:
            entry = entries[tr.scenario] = {
                "name": tr.scenario, "recorded": True, "grid_rows": tr.grid_rows,
                "agents": dict(tr.initial_state.get("agents", {})),
                "objects": object_positions(tr.initial_state.get("objects", {})),
                "description": (tr.level or {}).get("description", ""),
                "agent_names": [], "seeds": [], "episodes": 0, "loophole_episodes": 0, "flagged_episodes": 0}
        entry["episodes"] += 1
        entry["loophole_episodes"] += bool(tr.ground_truth_loophole_steps)
        entry["flagged_episodes"] += bool(tr.monitor_flags)
        for key, value in (("agent_names", tr.agent), ("seeds", tr.seed)):
            if value not in entry[key]:
                entry[key].append(value)
    if include_shipped:
        from gridlock.scenarios.loader import available_levels, load_scenario
        for name in available_levels():
            if name in entries:
                continue
            try:
                scenario = load_scenario(name)
            except Exception:                         # a broken level file shouldn't stop the viewer
                continue
            entries[name] = {"name": name, "recorded": False, "grid_rows": scenario.grid.to_rows(),
                             "agents": dict(scenario.agents), "objects": dict(scenario.grid.objects),
                             "description": scenario.description}
    return list(entries.values())
