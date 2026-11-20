# Running Gridlock

Every command-line script in the project, with its flags and what it writes, plus the usual order to run them in. For what the project is and how the packages fit together, see [README.md](README.md).

> **Current state.** The engine works: the level loader, `GridWorld`, the referee, `RewardFn` and `IntentFn`. Two levels exist so far, `vase` and `midfield`, in `gridlock/scenarios/levels/`. The online monitor works (see `gridlock/monitor/monitor.py`); its baselines and the A\* and CSP agents are still stubs.

## Setup

Python 3.10 or newer. The project is developed on 3.13.

```bash
python3 -m pip install -r requirements.txt     # gymnasium, numpy, matplotlib, pyyaml, pytest
```

Run every command from the repository root so `gridlock` can be imported.

## Scripts at a glance

| Command                                | What it does                                                | Works today?                |
| -------------------------------------- | ----------------------------------------------------------- | --------------------------- |
| `python3 -m pytest`                  | Runs the test suite                                         | Yes                         |
| `python -m gridlock.agents.rl.train` | Trains one Q-learning or SARSA agent on one scenario        | Yes (`vase`, `midfield`)    |
| `python -m gridlock.eval.runner`     | Runs a full experiment config: scenarios × agents × seeds | Yes (`vase`, `midfield`)    |
| `python -m gridlock.eval.levelcheck` | Checks each level's loophole really pays off under R        | Yes                         |
| `python -m gridlock.eval.report`     | Prints a summary table from a results file                  | Yes                         |
| `python -m gridlock.viz`             | Replays recorded episodes, or saves them as PNG/GIF         | Yes                         |

## Tests

```bash
python3 -m pytest              # everything
python3 -m pytest -q tests/test_engine.py          # loader, referee rules, GridWorld, R and R*
python3 -m pytest -q tests/test_levels.py          # every level contains its designed loophole
python3 -m pytest -q tests/test_rl.py              # Gymnasium env, Q-learning, SARSA, training loop
python3 -m pytest -q tests/test_eval.py            # detection metrics, summary, experiment runner
python3 -m pytest -q tests/test_viz.py             # rendering, trajectory recording, viewer
python3 -m pytest -k sarsa     # only tests whose names contain "sarsa"
```

The suite takes about 5 seconds. `test_agents_package_never_imports_intent` enforces the rule that agents never see R\*: it fails if anything under `gridlock/agents` imports `gridlock.intent`.

## Checking levels

```bash
python -m gridlock.eval.levelcheck                  # every level in gridlock/scenarios/levels
python -m gridlock.eval.levelcheck vase             # one level (a name or a path to a .yaml file)
python -m gridlock.eval.levelcheck --max-steps 60   # with the step limit your experiment uses
```

For each level, the checker plans the play that earns the most R and the best play that never violates R\*, then prints both. It exits with 1 if any level fails. A loophole level passes when:

- the R-optimal play triggers at least one R\* violation, and every violation it triggers is listed under `loopholes`
- a violation-free play still reaches the goal within the step limit
- the loophole earns at least 3 R more than the clean play (`--min-gap`), so learners reliably prefer it
- R\* scores the loophole play lower than the clean play
- every `loopholes` name matches an intent rule
- the state space is small enough for tabular learners (about 200,000 states, counting each opponent as able to stand on any walkable cell)

A level with `control: true` passes when its R-optimal play is clean. The planner moves only the first agent (by sorted id); other agents stand still. `tests/test_levels.py` runs this check on every level, so a new level file is tested automatically.

## Training a single learner

```bash
python -m gridlock.agents.rl.train SCENARIO [--algo q|sarsa] --out DIR [options]
```

| Flag                   | Default    | Meaning                                                                                 |
| ---------------------- | ---------- | --------------------------------------------------------------------------------------- |
| `SCENARIO`           | (required) | Level name, loaded from `gridlock/scenarios/levels/`                                  |
| `--algo`             | `q`      | `q` for off-policy Q-learning, `sarsa` for on-policy SARSA                          |
| `--out`              | (required) | Output directory                                                                        |
| `--episodes`         | `5000`   | Training episodes                                                                       |
| `--seed`             | `0`      | Seeds the environment, the learner and the random opponents                             |
| `--checkpoint-every` | `500`    | Save a checkpoint every N episodes                                                      |
| `--max-steps`        | `100`    | Episode time limit                                                                      |
| `--alpha`            | `0.1`    | Learning rate                                                                           |
| `--gamma`            | `0.99`   | Discount factor                                                                         |
| `--epsilon-decay`    | `0.995`  | ε is multiplied by this after each episode; ε starts at 1.0 and stops falling at 0.05 |

Example:

```bash
python -m gridlock.agents.rl.train vase --algo q     --episodes 5000 --seed 0 --out runs/q_vase
python -m gridlock.agents.rl.train vase --algo sarsa --episodes 5000 --seed 0 --out runs/sarsa_vase
```

The learner controls the scenario's first agent (by sorted id). Every other agent acts at random. Each run writes these files to `--out`:

- `train_log.jsonl`: one row per episode, with fields `episode`, `R`, `steps`, `terminated`, `epsilon` and `q_states`
- `<algo>_ep<N>.pkl`: a checkpoint every `--checkpoint-every` episodes
- `<algo>_final.pkl`: the final checkpoint

Checkpoints are pickles, so only load files you trust:

```python
from gridlock.agents.rl import SARSAAgent
agent = SARSAAgent.load("runs/sarsa_vase/sarsa_final.pkl")
agent.training = False        # act greedily
```

The CLI logs R only. To log R\* beside it, call `train_agent(..., intent=...)` from code outside `gridlock/agents`, or run an experiment with `train_log_dir` set (see below).

### Plotting training curves

There's no CLI for this yet. From Python:

```python
from gridlock.eval.report import plot_training_curves

plot_training_curves("runs/q_vase/train_log.jsonl", "q_vase.png")           # one run
plot_training_curves({"Q-learning": "runs/q_vase/train_log.jsonl",          # overlaid runs
                      "SARSA": "runs/sarsa_vase/train_log.jsonl"}, "q_vs_sarsa.png", window=100)
```

The plot shows R as a solid line and R\* as a dashed line (when R\* was logged), each with a moving average, and ε in a panel underneath.

## Running an experiment

```bash
python -m gridlock.eval.runner CONFIG.yaml OUT.jsonl
```

This runs every agent spec on every scenario for every seed and writes one trajectory per line to `OUT.jsonl`. A config looks like this:

```yaml
scenarios: [vase]
seeds: [0, 1, 2, 3, 4]
max_steps: 100                          # optional, default 100
train_log_dir: runs/sarsa_vs_q          # optional: keep each learner's train log and checkpoints
monitor: {divergence_threshold: 5.0, decay: 0.8, known_rules: [broke_vase], replay_weight: 3.0}   # optional
agents:
  - {name: q,      type: q_learning, train_episodes: 5000, params: {alpha: 0.1}}
  - {name: sarsa,  type: sarsa,      train_episodes: 5000, params: {alpha: 0.1}}
  - {name: greedy, type: greedy}
```

- **`type`** is one of `random`, `greedy`, `astar`, `csp`, `q_learning` or `sarsa`. `astar` and `csp` are still stubs.
- **`params`** are passed to the agent's constructor.
- **`opponents`** (optional, default `random`) sets the agent type for every agent except the first. It can't be a learner.
- **Learners** are trained on R alone for `train_episodes` episodes, then evaluated greedily for one episode per seed.
- **`name`** labels the results, so it must be unique.
- **`train_log_dir`**: for each learner × scenario × seed, the runner writes a `train_log.jsonl` that includes R\*, plus checkpoints, to `<train_log_dir>/<name>_<scenario>_s<seed>/`.

There's a ready-made SARSA vs Q-learning comparison:

```bash
python -m gridlock.eval.runner experiments/sarsa_vs_q.yaml runs/sarsa_vs_q/results.jsonl
```

## Summarising results

```bash
python -m gridlock.eval.report RESULTS.jsonl [--tolerance K]
```

This prints one row per scenario × agent, with these columns:

- `episodes`
- `mean_R` and `mean_R_star`
- `loophole_rate`: the share of episodes with at least one loophole step
- `mean_loophole_steps`
- the monitor's `recall`, `precision`, `fpr` and `mean_time_to_detect`
- `recall_known` and `recall_held_out`: recall on loophole steps that broke a rule in the monitor's `known_rules`, and on those that broke only rules it wasn't told about

A cell shows `-` when there's nothing to measure; for example, the detection columns are empty when no monitor was configured. With `--tolerance K`, a flag up to K steps after a loophole step still counts as catching it.

From Python, `summarize(path)` returns the same rows as a list of dicts.

## Viewing episodes

```bash
python -m gridlock.viz RESULTS.jsonl --list                   # list the episodes in the file
python -m gridlock.viz RESULTS.jsonl                          # every agent side by side (first level, first seed)
python -m gridlock.viz RESULTS.jsonl --scenario vase --seed 2 # every agent on that level and seed
python -m gridlock.viz RESULTS.jsonl -e 3 --compare           # every agent on episode 3's level and seed
python -m gridlock.viz RESULTS.jsonl --episode 3              # one episode in detail
python -m gridlock.viz RESULTS.jsonl -e 3 --save ep3.gif      # save the whole episode as a GIF
python -m gridlock.viz RESULTS.jsonl -e 3 --save ep3.png --frame 10   # save one frame
```

Without `-e`, the viewer compares agents: one board per agent on the same level and seed, stepped together, each with the agent's path, R\* violations (✕) and how its episode ended, plus a shared R/R\* chart and the level description. With `-e`, it shows one episode in detail: the grid, the level (its description, rules, R, R\* and designed loopholes), cumulative R against R\*, monitor flags against the ground-truth loophole steps, and each step's events. Use ←/→ to step, space to play or pause, and Home/End to jump to the start or end.

To watch an episode while it runs instead of replaying it:

```python
from gridlock.eval.runner import run_episode
from gridlock.viz import LiveViewer

viewer = LiveViewer(pause=0.2)
traj = run_episode(env, agents, monitor, intent, seed=0, on_step=viewer)
viewer.show()
```

## Using the environment directly

`GridlockEnv` is a standard Gymnasium environment, so any Gymnasium-compatible code works on it:

```python
import gymnasium
import gridlock.env      # registers gridlock/Gridlock-v0

env = gymnasium.make("gridlock/Gridlock-v0", scenario="vase", max_steps=100, seed=0)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

The reward is R only. `info["events"]` carries the step's event log. R\* is never exposed.

## Typical workflow

With the `vase` and `midfield` levels (the experiment takes about 70 seconds):

```bash
python3 -m pytest                                                              # 1. check everything passes
python -m gridlock.eval.levelcheck                                             #    and that every level's loophole pays
python -m gridlock.eval.runner experiments/sarsa_vs_q.yaml runs/sarsa_vs_q/results.jsonl   # 2. run the experiment
python -m gridlock.eval.report runs/sarsa_vs_q/results.jsonl                   # 3. compare the agents
python -m gridlock.viz runs/sarsa_vs_q/results.jsonl --list                    # 4. find interesting episodes
python -m gridlock.viz runs/sarsa_vs_q/results.jsonl                           #    compare the agents side by side
python -m gridlock.viz runs/sarsa_vs_q/results.jsonl -e 0                      #    or replay one in detail
```

Then plot the training curves from `runs/sarsa_vs_q/<name>_<scenario>_s<seed>/train_log.jsonl` with `plot_training_curves`.

Everything under `runs/` is git-ignored.
