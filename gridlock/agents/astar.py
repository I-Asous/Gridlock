"""A* planner: the play with the most total R within the step limit (cost = -R per transition).

The search runs over (state, time) pairs, where a state is the full GridWorld state with its clock
zeroed. Every other agent is modelled as standing still (see gridlock.agents.planning).

Making the costs fit A*. R can be positive (goal, bonus points), so -R is a negative cost, and a
bonus that pays on every visit makes a negative cycle. Both go away once every play is made the
same length H, the step limit:

- a transition costs M - r, where M is an upper bound on one step's R, so no cost is negative;
- reaching the goal ends the play, and the remaining H - t steps are padded at reward 0, which
  costs M per step. Running out of time ends the play at no cost.

Every complete play then costs M * H - R, so the cheapest play is the one with the most R. Time
only moves forward, so the search space is finite and A* terminates.

Heuristic. The best R still available bounds the remaining cost from below. With s the reward
per step, g the goal reward and d the walking distance to the nearest goal (ignoring objects,
traps and other agents), no step that doesn't reach the goal earns more than p = M - g. Reaching
the goal after k >= d steps earns at most k * p + g, and never reaching it earns at most
(H - t) * p. The heuristic is M * (H - t) minus the larger of those bounds. It is admissible, so
the first time the end of the play is popped from the queue, that play is optimal.

M comes from the reward config: the step reward plus every positive event weight. If a
transition ever earns more than the bound assumes (several events of one kind in one step, or a
reward function without a config), the search restarts with a larger M, which keeps it exact.
"""

import heapq
from dataclasses import replace
from itertools import count

from gridlock.agents.planning import PlanningAgent, untimed
from gridlock.env.events import EventKind
from gridlock.env.grid import Action, Tile

END = "end"   # search node for "the play is over"


class _BoundTooLow(Exception):
    def __init__(self, needed: float):
        self.needed = needed


class AStarAgent(PlanningAgent):
    def __init__(self, agent_id: str, max_expansions: int = 1_000_000):
        super().__init__(agent_id)
        self.max_expansions = max_expansions
        self.expansions = 0                    # nodes expanded by the last search
        self._successors: dict = {}            # untimed state -> [(action, untimed next state, R)]
        self._goal_distance: dict = {}
        self._horizon = 0
        self._bound = 0.0                      # M: no step earns more R than this
        self._goal_reward = 0.0                # g

    def make_plan(self, env) -> list[Action]:
        return self.search(env)

    def reset(self, env) -> None:
        self._successors = {}
        self._goal_distance = env.grid.distances_from(env.grid.cells(Tile.GOAL))
        super().reset(env)

    def search(self, env) -> list[Action]:
        """The best plan from env.get_state() to the end of the episode. When the expansion budget
        runs out, the plan to the best frontier node found so far (by f) is returned instead."""
        self._horizon = getattr(env, "max_steps", 100)
        self._bound, self._goal_reward = _reward_bounds(env.reward_fn)
        while True:
            try:
                return self._astar(env)
            except _BoundTooLow as raised:
                self._bound = raised.needed

    def heuristic(self, state, env) -> float:
        """A lower bound on the cost from `state` (whose t is the time) to the end of the play."""
        left = max(0, self._horizon - state.t)
        if self.agent_id in state.done:
            return self._bound * left            # exact: padding to the step limit
        p, g = self._bound - self._goal_reward, self._goal_reward
        best = left * p
        d = self._goal_distance.get(dict(state.agents)[self.agent_id])
        if d is not None and max(d, 1) <= left:           # the goal can still be reached
            best = max(best, (left if p > 0 else max(d, 1)) * p + g)
        return self._bound * left - best

    # --- search --------------------------------------------------------------------------------

    def _astar(self, env) -> list[Action]:
        start = env.get_state()
        t0 = start.t
        root = (untimed(start), t0)
        tie = count()
        best_cost = {root: 0.0}
        parent: dict = {root: None}
        frontier = [(self.heuristic(start, env), next(tie), 0.0, root)]
        self.expansions = 0
        while frontier:
            f, _, cost, node = heapq.heappop(frontier)
            if node == END:
                return self._actions(parent)
            if cost > best_cost.get(node, float("inf")):
                continue                         # stale entry
            self.expansions += 1
            if self.expansions > self.max_expansions:
                return self._actions(parent, node)
            state, t = node
            for step_cost, child, action in self._edges(env, state, t):
                new_cost = cost + step_cost
                if new_cost < best_cost.get(child, float("inf")):
                    best_cost[child] = new_cost
                    parent[child] = (node, action)
                    h = 0.0 if child == END else self.heuristic(replace(child[0], t=child[1]), env)
                    heapq.heappush(frontier, (new_cost + h, next(tie), new_cost, child))
        return []

    def _edges(self, env, state, t):
        """(cost, child node, action) for each way on from `state` at time t."""
        if self.agent_id in state.done:
            return [(self._bound * (self._horizon - t), END, None)]
        if t >= self._horizon:
            return [(0.0, END, None)]
        edges = []
        for action, nxt, r in self._expand(env, state):
            reaches_goal = self.agent_id in nxt.done
            limit = self._bound if reaches_goal else self._bound - self._goal_reward
            if r > limit + 1e-9:
                raise _BoundTooLow(r + (self._goal_reward if not reaches_goal else 0.0))
            edges.append((self._bound - r, (nxt, t + 1), action))
        return edges

    def _expand(self, env, state):
        if state not in self._successors:
            out = []
            for action in env.legal_actions(state, self.agent_id):
                nxt, rewards, _ = env.simulate(state, {self.agent_id: action})
                out.append((action, untimed(nxt), rewards[self.agent_id]))
            self._successors[state] = out
        return self._successors[state]

    @staticmethod
    def _actions(parent: dict, node=END) -> list[Action]:
        actions = []
        while parent.get(node) is not None:
            node, action = parent[node]
            if action is not None:
                actions.append(action)
        return actions[::-1]


def _reward_bounds(reward_fn) -> tuple[float, float]:
    """(M, g): an upper bound on one step's R, and the goal reward, from a RewardFn's config.
    A reward function without one starts at (0, 0); the search raises M as it needs to."""
    weights = getattr(reward_fn, "weights", None)
    step = float(getattr(reward_fn, "step_reward", 0.0))
    if weights is None:
        return 0.0, 0.0
    goal = max(0.0, float(weights.get(EventKind.REACHED_GOAL, 0.0)))
    return step + sum(max(0.0, float(w)) for w in weights.values()), goal
