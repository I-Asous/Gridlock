# Gridlock

A multi-agent gridworld testbed for studying **specification gaming**: agents that maximise the reward they were given while violating what the designer actually wanted.

## Goal

Every scenario has two reward signals:

- **R**, the stated reward. Agents see it and optimise it (step costs, checkpoints, reaching the goal).
- **R\***, the hidden designer intent. It covers things like "don't break the vase" or "don't switch off the sensor". Agents never see it. Only the monitor and the evaluation code read it.

Each level contains deliberate loopholes, where R can be pushed up in ways that R\* penalises. The project aims to:

1. **Elicit gaming.** Run several kinds of agents on these levels and measure how often, and how badly, each one exploits the gap between R and R\*:
   - planners: A\* and CSP
   - learners: tabular Q-learning, with DQN as a fallback
   - scripted baselines: random and greedy
2. **Detect it online.** Build a monitor that watches the event stream step by step and flags an agent as soon as its behaviour diverges from intent, without waiting for the episode to end.
3. **Evaluate detection.** Compare the online monitor against non-ML baselines (an end-of-episode audit and a revisit/path-overrun heuristic) using:
   - **Gaming gap:** normalised R minus normalised R\*. An episode counts as gaming when the gap exceeds δ.
   - **Detection rate and precision:** recall and precision of flags against the ground-truth loophole steps.
   - **False-positive rate**
   - **Time-to-detect:** the first flag step minus the first loophole step.

## Layout

| Package | Purpose |
| --- | --- |
| `gridlock/env` | Grid, tiles, actions, the typed event log, the referee (legality and multi-agent conflicts), and the Gym-style `GridWorld` |
| `gridlock/rules` | `RewardFn`, the stated reward R |
| `gridlock/intent` | `IntentFn`, the hidden intent R\*. **Agents must never import this package.** |
| `gridlock/agents` | A\*, CSP, scripted agents, and RL agents in `agents/rl` |
| `gridlock/monitor` | The online R\*-aware monitor and the baseline monitors |
| `gridlock/scenarios` | YAML level loader and levels with designed loopholes |
| `gridlock/eval` | Experiment runner, metrics, and reports |

## Status

The scaffolding is in place: interfaces, data types and module layout. Most of the logic is still `NotImplementedError` stubs.
