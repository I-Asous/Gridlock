"""R: the stated reward agents optimise. Configured per scenario."""

from gridlock.env.events import Event, EventKind

# Friendly config keys for event kinds; any EventKind name in lower case also works (e.g. "bumped").
ALIASES = {"goal": EventKind.REACHED_GOAL}


def event_kind(key: str) -> EventKind:
    if key in ALIASES:
        return ALIASES[key]
    try:
        return EventKind[key.upper()]
    except KeyError:
        raise ValueError(f"unknown reward key {key!r}; use 'step', 'goal' or an event kind such as "
                         f"{', '.join(kind.name.lower() for kind in EventKind)}") from None


def finished_before(state, events: list[Event], agent_id: str) -> bool:
    """True when the agent had already reached the goal before this step, so it no longer earns
    anything. `state` is the state after the step."""
    done = getattr(state, "done", frozenset())
    return agent_id in done and not any(e.agent_id == agent_id and e.kind is EventKind.REACHED_GOAL
                                        for e in events)


class RewardFn:
    """Per-step reward: `step` every step the agent is still playing, plus a weight for each of the
    agent's own events this step, e.g. {"step": -1, "goal": 10, "checkpoint": 1, "bumped": -0.5}."""

    def __init__(self, config: dict):
        self.config = config   # e.g. {"step": -1, "goal": 10, "checkpoint": 1}
        self.step_reward = float(config.get("step", 0.0))
        self.weights = {event_kind(key): float(value) for key, value in config.items() if key != "step"}

    def __call__(self, state, events: list[Event], agent_id: str) -> float:
        if finished_before(state, events, agent_id):
            return 0.0
        total = self.step_reward
        for event in events:
            if event.agent_id == agent_id:
                total += self.weights.get(event.kind, 0.0)
        return total
