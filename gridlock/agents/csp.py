"""CSP planner: pos/object variables over horizon T, solved with backtracking + MRV + AC-3.

The planner finds the shortest plan that reaches the goal, trying horizons T = d, d + 1, ... up
to `horizon` (d is the walking distance to the goal). It is a satisfaction planner: R is never
read. That makes it a useful contrast with A*, which maximises R. The CSP takes the quickest legal
way to the goal, loophole or not, but it never farms reward along the way.

Variables, for t = 0..T:
    P_t  the agent's cell after t steps
    O_t  the object layout after t steps: one entry per object, a cell or None once broken

Domains, pruned up front (node consistency):
    P_t  walkable cells not held by another agent, no closer to the start than t steps allow and
         no further from a goal than T - t steps allow. P_T must be a goal and earlier P_t must not.
    O_t  layouts where every object is where it is now, on a floor cell (if pushable) or broken (if
         breakable), with at most t pushes or breaks between them and the current layout.

Constraints:
    binary  P_t and P_t+1 are the same or neighbouring cells (or P_t+1 is the trap return cell);
            O_t and O_t+1 differ by at most one object, pushed one cell or broken;
            P_t is not on an object in O_t.
    4-ary   (P_t, O_t) -> (P_t+1, O_t+1) is a transition the rules allow for some action. This is
            checked by simulating the GridWorld, so the plan respects every referee rule (pushing,
            breaking, traps, bumping into an agent that stands still).

AC-3, generalised to n-ary constraints (GAC-3), runs before the search and after every
assignment (maintaining arc consistency), so each transition prunes positions and layouts both
forward and backward in time. Variables are picked by MRV, with ties broken by degree and then by
time.

extra_constraints forbid transitions, for ablations such as "plan as if you knew R*". Each one is
either a dict in the format of an intent `forbid` rule, {kind, target?, payload?} (any other keys
such as name and penalty are ignored), which forbids every step that emits a matching event, or a
callable (state, action, next_state, events) -> bool that returns True for allowed steps. Dicts
can come straight from an experiment YAML, so this module never needs to import gridlock.intent.
"""

from collections import defaultdict, deque
from dataclasses import replace
from itertools import product
from typing import Callable, Hashable, Optional

from gridlock.agents.planning import PlanningAgent
from gridlock.env.grid import Action, Tile

DEFAULT_OBJECT = {"pushable": True, "breakable": False}   # as in gridlock.env.referee




class Problem:
    """Variables with finite domains, binary constraints, and n-ary constraints. Values must be
    hashable."""

    def __init__(self):
        self.domains: dict[Hashable, list] = {}
        self.binary: dict[Hashable, dict[Hashable, list[Callable]]] = defaultdict(dict)
        self.nary: list[tuple[tuple, Callable, Optional[Callable]]] = []
        self.nary_of: dict[Hashable, list[int]] = defaultdict(list)

    def add_variable(self, var: Hashable, domain) -> None:
        self.domains[var] = list(domain)

    def add_binary(self, x: Hashable, y: Hashable, pred: Callable) -> None:
        """pred(value of x, value of y) -> bool."""
        self.binary[x].setdefault(y, []).append(pred)
        self.binary[y].setdefault(x, []).append(lambda b, a: pred(a, b))

    def add_nary(self, scope: tuple, pred: Callable, tuples: Optional[Callable] = None) -> None:
        """pred(*values in scope order) -> bool. `tuples(*domains in scope order)`, if given,
        yields exactly the satisfying tuples drawn from those domains; it saves enumerating the
        whole product when the constraint can list its supports directly."""
        self.nary.append((scope, pred, tuples))
        for var in scope:
            self.nary_of[var].append(len(self.nary) - 1)


class SearchBudgetExceeded(Exception):
    pass


class Solver:
    """Backtracking search with MRV (ties: degree, then declaration order), maintaining
    generalised arc consistency (GAC-3) after every assignment."""

    def __init__(self, problem: Problem, max_nodes: int = 20_000):
        self.p = problem
        self.max_nodes = max_nodes
        self.nodes = 0
        self.order = {var: i for i, var in enumerate(problem.domains)}
        self.degree = {var: len(problem.binary[var]) + len(problem.nary_of[var]) for var in problem.domains}

    def solve(self) -> Optional[dict]:
        """A complete consistent assignment, or None when there is none. Raises
        SearchBudgetExceeded after max_nodes search nodes."""
        domains = {var: list(values) for var, values in self.p.domains.items()}
        work = [("arc", x, y) for x in self.p.binary for y in self.p.binary[x]]
        work += [("nary", k) for k in range(len(self.p.nary))]
        if not self.ac3(domains, work):
            return None
        return self._backtrack({}, domains)

    def ac3(self, domains: dict, work) -> bool:
        """Make every queued arc ("arc", x, y) and n-ary constraint ("nary", k) consistent, and
        requeue whatever a removal touches. False when a domain empties. Domains are replaced,
        never mutated, so callers can copy the dict shallowly."""
        queue = deque(work)
        queued = set(queue)
        while queue:
            item = queue.popleft()
            queued.discard(item)
            changed = self._revise(domains, item[1], item[2]) if item[0] == "arc" else self._revise_nary(domains, item[1])
            if changed is None:
                return False
            for var in changed:
                for follow in self._watchers(var, item):
                    if follow not in queued:
                        queue.append(follow)
                        queued.add(follow)
        return True

    def _watchers(self, var, cause):
        for z in self.p.binary[var]:
            if cause != ("arc", var, z):
                yield ("arc", z, var)
        for k in self.p.nary_of[var]:
            if cause != ("nary", k):
                yield ("nary", k)

    def _revise(self, domains: dict, x, y) -> Optional[list]:
        """Drop values of x with no support in y. Returns the changed variables ([] or [x]), or
        None when x's domain empties."""
        preds = self.p.binary[x][y]
        ys = domains[y]
        keep = [a for a in domains[x] if any(all(pred(a, b) for pred in preds) for b in ys)]
        if len(keep) == len(domains[x]):
            return []
        domains[x] = keep
        return [x] if keep else None

    def _revise_nary(self, domains: dict, k: int) -> Optional[list]:
        """Keep only values that appear in some satisfying tuple of constraint k."""
        scope, pred, tuples = self.p.nary[k]
        lists = [domains[v] for v in scope]
        rows = tuples(*lists) if tuples is not None else (row for row in product(*lists) if pred(*row))
        supported = [set() for _ in scope]
        for row in rows:
            for seen, value in zip(supported, row):
                seen.add(value)
        changed = []
        for var, values, seen in zip(scope, lists, supported):
            if len(seen) < len(values):
                keep = [a for a in values if a in seen]
                if not keep:
                    return None
                domains[var] = keep
                changed.append(var)
        return changed

    def _backtrack(self, assignment: dict, domains: dict) -> Optional[dict]:
        if len(assignment) == len(domains):
            return dict(assignment)
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise SearchBudgetExceeded
        var = min((v for v in domains if v not in assignment),
                  key=lambda v: (len(domains[v]), -self.degree[v], self.order[v]))
        for value in domains[var]:
            trial = dict(domains)
            trial[var] = [value]
            assignment[var] = value
            if self.ac3(trial, self._watchers(var, None)):
                result = self._backtrack(assignment, trial)
                if result is not None:
                    return result
            del assignment[var]
        return None


# --- the planner -----------------------------------------------------------------------------------

class CSPAgent(PlanningAgent):
    def __init__(self, agent_id: str, horizon: int = 30, extra_constraints: list | None = None,
                 max_nodes: int = 20_000):
        super().__init__(agent_id)
        self.horizon = horizon                               # longest plan tried
        self.extra_constraints = extra_constraints or []   # e.g. R*-derived constraints for ablation
        self.max_nodes = max_nodes                           # search nodes per horizon
        self._allowed = [_as_check(spec) for spec in self.extra_constraints]
        self._model: Optional[_Model] = None

    def make_plan(self, env) -> list[Action]:
        return self.solve(env)

    def build_problem(self, env, horizon: Optional[int] = None) -> Optional[Problem]:
        """The CSP for reaching a goal in exactly `horizon` steps from env.get_state() (default:
        the shortest horizon worth trying), or None when a domain is empty from the start."""
        model = self._model if self._model is not None and self._model.base is env.get_state() else \
            _Model(env, self.agent_id, self._allowed)
        self._model = model
        return model.problem(model.min_horizon if horizon is None else horizon)

    def solve(self, env) -> list[Action]:
        """The shortest plan to a goal within `horizon` steps (and the steps left in the
        episode), or [] when there is none. A horizon whose search runs past max_nodes is skipped."""
        self._model = _Model(env, self.agent_id, self._allowed)
        model = self._model
        if model.min_horizon is None:
            return []
        longest = min(self.horizon, getattr(env, "max_steps", self.horizon) - env.get_state().t)
        for T in range(model.min_horizon, longest + 1):
            problem = model.problem(T)
            if problem is None:
                continue
            try:
                solution = Solver(problem, self.max_nodes).solve()
            except SearchBudgetExceeded:
                continue
            if solution is not None:
                return model.actions(solution, T)
        return []


class _Model:
    """Everything the CSP needs about the world, read once per plan."""

    def __init__(self, env, agent_id: str, allowed: list[Callable]):
        self.env, self.me, self.allowed = env, agent_id, allowed
        self.base = env.get_state()
        grid = env.grid
        self.grid = grid
        agents = dict(self.base.agents)
        self.pos0 = tuple(agents[agent_id])
        self.blocked = {tuple(pos) for aid, pos in agents.items() if aid != agent_id and aid not in self.base.done}
        self.oids = [oid for oid, _ in self.base.objects]
        self.layout0 = tuple(_cell(value) for _, value in self.base.objects)
        self.goals = set(grid.cells(Tile.GOAL))
        start = tuple(getattr(env.scenario, "agents", {}).get(agent_id, self.pos0))
        self.trap_return = start if grid.cells(Tile.TRAP) else None
        self.to_goal = grid.distances_from(self.goals)
        from_here = grid.distances_from([self.pos0])
        from_start = grid.distances_from([start])
        # Steps needed to stand on each cell: walk there, or be sent back to the start by a trap
        # (which takes at least one step) and walk from there.
        self.reach = {c: min(from_here.get(c, 10**9), 1 + from_start.get(c, 10**9) if self.trap_return else 10**9)
                      for c in set(from_here) | set(from_start)}
        done = agent_id in self.base.done
        d = self.to_goal.get(self.pos0)
        self.min_horizon = None if done or not self.goals or d is None else max(d, 1)
        self._successors: dict = {}
        self._layouts = self._all_layouts(getattr(env.scenario, "object_props", {}) or {})

    # domains and constraints 
    def _all_layouts(self, props: dict) -> list[tuple[tuple, int]]:
        """(layout, pushes/breaks needed to get there) for every object layout worth considering."""
        floor = [c for c in self.grid.cells(Tile.FLOOR) if c not in self.blocked]
        options = []
        for oid, now in zip(self.oids, self.layout0):
            kind = {**DEFAULT_OBJECT, **props.get(oid, {})}
            values = [now]
            if now is not None:
                if kind.get("pushable"):
                    values += [c for c in floor if c != now]
                if kind.get("breakable"):
                    values.append(None)
            options.append(values)
        layouts = []
        for layout in product(*options):
            cells = [c for c in layout if c is not None]
            if len(cells) == len(set(cells)):
                layouts.append((layout, sum(_moves(a, b) for a, b in zip(self.layout0, layout))))
        layouts.sort(key=lambda item: (item[1], [(c is None, c or (0, 0)) for c in item[0]]))
        return layouts

    def problem(self, T: int) -> Optional[Problem]:
        p = Problem()
        cells = sorted((c for c in self.reach if c not in self.blocked and self.grid.is_walkable(c)),
                       key=lambda c: (self.to_goal.get(c, 10**9), c))
        for t in range(T + 1):
            if t == 0:
                positions = [self.pos0]
            else:
                positions = [c for c in cells if self.reach[c] <= t and self.to_goal.get(c, 10**9) <= T - t
                             and ((c in self.goals) == (t == T))]
            p.add_variable(("P", t), positions)
            p.add_variable(("O", t), [layout for layout, cost in self._layouts if cost <= t])
            if not positions or not p.domains[("O", t)]:
                return None
        for t in range(T + 1):
            p.add_binary(("P", t), ("O", t), _not_on_object)
        for t in range(T):
            p.add_binary(("P", t), ("P", t + 1), self._step_ok)
            p.add_binary(("O", t), ("O", t + 1), _layout_step_ok)
            p.add_nary((("P", t), ("O", t), ("P", t + 1), ("O", t + 1)), self._transition_ok, self._transitions)
        return p

    def _step_ok(self, a, b) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1 or b == self.trap_return

    def _transition_ok(self, pos, layout, next_pos, next_layout) -> bool:
        return (next_pos, next_layout) in self._next(pos, layout)

    def _transitions(self, positions, layouts, next_positions, next_layouts):
        """Every allowed (P_t, O_t, P_t+1, O_t+1) drawn from the given domains."""
        next_positions, next_layouts = set(next_positions), set(next_layouts)
        for pos in positions:
            for layout in layouts:
                if pos in layout:
                    continue
                for next_pos, next_layout in self._next(pos, layout):
                    if next_pos in next_positions and next_layout in next_layouts:
                        yield pos, layout, next_pos, next_layout

    # rules

    def _state(self, pos, layout):
        agents = tuple((aid, pos if aid == self.me else p) for aid, p in self.base.agents)
        objects = tuple((oid, value) for oid, value in zip(self.oids, layout))
        return replace(self.base, t=0, agents=agents, objects=objects, done=self.base.done - {self.me})

    def _next(self, pos, layout) -> dict:
        """(next cell, next layout) -> the first action that gets there, over allowed steps."""
        key = (pos, layout)
        if key not in self._successors:
            state = self._state(pos, layout)
            out = {}
            for action in self.env.legal_actions(state, self.me):
                nxt, _, events = self.env.simulate(state, {self.me: action})
                if all(check(state, action, nxt, events) for check in self.allowed):
                    out.setdefault((tuple(dict(nxt.agents)[self.me]), tuple(_cell(v) for _, v in nxt.objects)),
                                   action)
            self._successors[key] = out
        return self._successors[key]

    def actions(self, solution: dict, T: int) -> list[Action]:
        steps = []
        for t in range(T):
            here = (solution[("P", t)], solution[("O", t)])
            there = (solution[("P", t + 1)], solution[("O", t + 1)])
            steps.append(self._next(*here)[there])
        return steps


def _cell(value):
    return None if value is None else tuple(value)


def _moves(a, b) -> int:
    """Pushes or breaks needed to take one object from a to b."""
    if a == b:
        return 0
    if a is None:
        return 10**9                      # a broken object stays broken
    return 1 if b is None else abs(a[0] - b[0]) + abs(a[1] - b[1])


def _not_on_object(pos, layout) -> bool:
    return pos not in layout


def _layout_step_ok(a: tuple, b: tuple) -> bool:
    """At most one object changes in one step: pushed to a neighbouring cell, or broken."""
    changed = [(x, y) for x, y in zip(a, b) if x != y]
    if not changed:
        return True
    if len(changed) > 1:
        return False
    (x, y), = changed
    return x is not None and (y is None or abs(x[0] - y[0]) + abs(x[1] - y[1]) == 1)


def _as_check(spec) -> Callable:
    """An extra constraint as a callable (state, action, next_state, events) -> allowed."""
    if callable(spec):
        return spec
    if not isinstance(spec, dict) or "kind" not in spec:
        raise ValueError(f"extra constraint {spec!r}: use a callable or a dict like {{kind: BROKE, target: vase}}")
    kind, target, payload = str(spec["kind"]).upper(), spec.get("target"), spec.get("payload") or {}

    def matches(event) -> bool:
        return (event.kind.name == kind and (target is None or event.target == target)
                and all(event.payload.get(key) == value for key, value in payload.items()))

    return lambda state, action, next_state, events: not any(matches(event) for event in events)
