"""Rule referee: legality checks, move application, multi-agent conflict resolution.

The rules every agent plays by (a loophole is anything these allow that the intent forbids):

- Moves are simultaneous. Moving into a wall or off the board fails (BUMPED, target "wall").
- Two agents moving into the same cell both stay, as do two agents trying to swap cells. Moving
  into a cell whose occupant stays put fails too. Following an agent into the cell it is leaving
  is allowed. Each agent that is stopped gets BUMPED (target: the other agent) and the other
  agent gets BLOCKED (target: the agent it stopped).
- Agents that have reached the goal are finished: they no longer act or block anyone.
- Moving into an object pushes it one cell (PUSHED) when it is pushable and the cell beyond is
  free floor: no wall, no special tile, no object, no agent. Otherwise a breakable object breaks
  (BROKE) and is removed, with the agent staying where it was; anything else is a BUMPED.
- INTERACT on a switch tile toggles that switch's flag (TOGGLED, payload {"on": new value}).
  INTERACT anywhere else does nothing.
- Entering a goal tile ends the agent's episode (REACHED_GOAL). Entering a checkpoint tile emits
  CHECKPOINT every time, with payload {"first": True} only on that agent's first visit to it.
- Entering a trap tile emits TRAPPED and sends the agent back to its start cell (payload {"to": start}).
  If another agent ends the step on that start cell, the trapped agent stays on the trap instead.
  While any of the level's `trap_off_flags` is on (a switch declared with `disables: traps`),
  traps do nothing and are walked over like floor.
"""

from collections import defaultdict
from dataclasses import replace
from typing import Optional

from gridlock.env.events import Event, EventKind, Pos
from gridlock.env.grid import Action, Grid, Tile

DEFAULT_OBJECT = {"pushable": True, "breakable": False}


def _shift(pos: Pos, action: Action) -> Pos:
    dx, dy = action.value
    return pos[0] + dx, pos[1] + dy


def _is_move(action: Action) -> bool:
    return action.value is not None and action.value != (0, 0)


class Referee:
    def __init__(self, grid: Grid, object_props: Optional[dict[str, dict]] = None,
                 switches: Optional[dict[Pos, str]] = None, starts: Optional[dict[str, Pos]] = None,
                 trap_off_flags=()):
        self.grid = grid
        self.object_props = dict(object_props or {})
        self.switches = dict(switches or {})
        self.starts = {aid: tuple(pos) for aid, pos in (starts or {}).items()}
        self.trap_off_flags = frozenset(trap_off_flags or ())

    def legal_actions(self, state, agent_id: str) -> list[Action]:
        """Actions that can do something: STAY, moves that don't walk into a wall (pushing or
        breaking an object counts) and INTERACT on a switch. A finished agent can only STAY."""
        if agent_id in state.done:
            return [Action.STAY]
        pos = dict(state.agents)[agent_id]
        actions = []
        for action in Action:
            if action is Action.INTERACT:
                if pos in self.switches:
                    actions.append(action)
            elif not _is_move(action) or self.grid.is_walkable(_shift(pos, action)):
                actions.append(action)
        return actions

    def resolve(self, state, actions: dict[str, Action], t: int) -> tuple[object, list[Event]]:
        """Apply joint actions simultaneously; return (next_state, events). Agents missing from
        `actions` stay put. `state` is not modified."""
        pos = dict(state.agents)
        objects = dict(state.objects)
        active = [aid for aid in sorted(pos) if aid not in state.done]
        object_at = {tuple(p): oid for oid, p in objects.items() if p is not None}
        events: list[Event] = []
        dest = dict(pos)
        pushes: dict[str, tuple[str, Pos]] = {}      # agent -> (object, cell the object moves to)
        breaks: dict[str, str] = {}                  # agent -> object it breaks

        def stop(aid: str, target: str, blocker: Optional[str] = None) -> None:
            dest[aid] = pos[aid]
            pushes.pop(aid, None)
            events.append(Event(t, aid, EventKind.BUMPED, pos[aid], target=target))
            if blocker is not None:
                events.append(Event(t, blocker, EventKind.BLOCKED, pos[blocker], target=aid))

        # 1. Each agent's intended move, against walls and objects.
        for aid in active:
            action = actions.get(aid, Action.STAY)
            if not _is_move(action):
                continue
            target = _shift(pos[aid], action)
            if not self.grid.is_walkable(target):
                stop(aid, "wall")
            elif target in object_at:
                oid = object_at[target]
                beyond = _shift(target, action)
                props = self.object_props.get(oid, DEFAULT_OBJECT)
                if props.get("pushable") and self._object_can_enter(beyond, object_at, pos, active):
                    pushes[aid] = (oid, beyond)
                    dest[aid] = target
                elif props.get("breakable"):
                    breaks[aid] = oid
                else:
                    stop(aid, oid)
            else:
                dest[aid] = target

        # 2. Pushes: an object being broken can't be pushed, and two pushes can't share an object
        #    or a destination cell. A cell an object is pushed into is taken for everyone else.
        broken = set(breaks.values())
        by_object, by_cell = defaultdict(list), defaultdict(list)
        for aid, (oid, beyond) in pushes.items():
            by_object[oid].append(aid)
            by_cell[beyond].append(aid)
        for aid, (oid, beyond) in list(pushes.items()):
            if oid in broken or len(by_object[oid]) > 1 or len(by_cell[beyond]) > 1:
                stop(aid, oid)
        claimed = {beyond: oid for oid, beyond in pushes.values()}
        for aid in active:
            if aid not in pushes and dest[aid] != pos[aid] and dest[aid] in claimed:
                stop(aid, claimed[dest[aid]])

        # 3. Agent conflicts, repeated until stable because a stopped agent can stop the one behind it.
        changed = True
        while changed:
            changed = False
            wanting = defaultdict(list)
            for aid in active:
                wanting[dest[aid]].append(aid)
            for ids in wanting.values():
                if len(ids) > 1:
                    for aid in ids:
                        if dest[aid] != pos[aid]:
                            other = next(b for b in ids if b != aid)
                            stop(aid, other, other)
                            changed = True
            if changed:
                continue
            occupant = {pos[aid]: aid for aid in active}
            for aid in active:
                other = occupant.get(dest[aid])
                if dest[aid] == pos[aid] or other is None:
                    continue
                if dest[other] == pos[other] or dest[other] == pos[aid]:   # stays put, or a swap
                    stop(aid, other, other)
                    changed = True

        # 4. Apply what survived.
        new_objects = dict(objects)
        for aid, oid in breaks.items():
            events.append(Event(t, aid, EventKind.BROKE, tuple(objects[oid]), target=oid))
            new_objects[oid] = None
        flags, done, visited = set(state.flags), set(state.done), set(state.visited)
        trapped = []
        for aid in active:
            if dest[aid] != pos[aid]:
                events.append(Event(t, aid, EventKind.MOVED, dest[aid]))
                if aid in pushes:
                    oid, beyond = pushes[aid]
                    new_objects[oid] = beyond
                    events.append(Event(t, aid, EventKind.PUSHED, beyond, target=oid,
                                        payload={"from": list(dest[aid])}))
                tile = self.grid.tile_at(dest[aid])
                if tile is Tile.GOAL:
                    done.add(aid)
                    events.append(Event(t, aid, EventKind.REACHED_GOAL, dest[aid]))
                elif tile is Tile.CHECKPOINT:
                    first = (aid, dest[aid]) not in visited
                    visited.add((aid, dest[aid]))
                    events.append(Event(t, aid, EventKind.CHECKPOINT, dest[aid], payload={"first": first}))
                elif tile is Tile.TRAP and not (self.trap_off_flags & state.flags):
                    trapped.append(aid)
            elif actions.get(aid) is Action.INTERACT and pos[aid] in self.switches:
                flag = self.switches[pos[aid]]
                on = flag not in flags
                (flags.add if on else flags.discard)(flag)
                events.append(Event(t, aid, EventKind.TOGGLED, pos[aid], target=flag, payload={"on": on}))

        for aid in trapped:
            start = self.starts.get(aid, pos[aid])
            taken = any(dest[other] == start for other in active if other != aid)
            events.append(Event(t, aid, EventKind.TRAPPED, dest[aid], payload={"to": list(dest[aid] if taken else start)}))
            if not taken:
                dest[aid] = start

        events.sort(key=lambda e: e.agent_id)        # stable: each agent's events stay in order
        next_state = replace(state, t=t,
                             agents=tuple((aid, dest[aid]) for aid, _ in state.agents),
                             objects=tuple((oid, new_objects[oid]) for oid, _ in state.objects),
                             flags=frozenset(flags), done=frozenset(done), visited=frozenset(visited))
        return next_state, events

    def _object_can_enter(self, cell: Pos, object_at: dict[Pos, str], pos: dict[str, Pos],
                          active: list[str]) -> bool:
        return (self.grid.in_bounds(cell) and self.grid.tile_at(cell) is Tile.FLOOR and cell not in object_at
                and all(pos[aid] != cell for aid in active))
