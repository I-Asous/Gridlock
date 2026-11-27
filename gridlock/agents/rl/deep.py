"""Deep RL learners backed by Stable-Baselines3: DQN, PPO and A2C. Trained on R only.

They train on the same GridlockEnv as the tabular learners, seen through GridImageObservation
(see gridlock.agents.rl.grid_image), and act like any other gridlock Agent once trained, so the
runner, the monitor and the viewer treat them like the rest.

    agent = DQNAgent("a0", seed=0, policy="cnn")
    rows = agent.train(GridlockEnv(GridWorld(scenario, max_steps=100), "a0"), steps=50_000, out_dir="runs/dqn_vase")
    action = agent.act(obs)          # obs: the usual observation vector

`policy` is "mlp" (the image flattened into an MLP, SB3's default) or "cnn" (GridCNN, which
shares weights across cells). Any other keyword goes to the SB3 algorithm, e.g. learning_rate.

Stable-Baselines3 and torch are optional: they are imported when an agent is built, so the rest
of gridlock works without them (`pip install -r requirements-deep.txt`).
"""

import contextlib
import json
from pathlib import Path
from typing import Optional

import gymnasium

from gridlock.agents.base import Agent
from gridlock.agents.rl.grid_image import GridImage, GridImageObservation
from gridlock.env.gym_env import ACTIONS


def _sb3():
    try:
        import stable_baselines3
    except ImportError:
        raise ImportError("deep RL agents need Stable-Baselines3 and torch: "
                          "pip install -r requirements-deep.txt") from None
    return stable_baselines3


class DeepAgent(Agent):
    """Common code for the SB3-backed agents. Subclasses set `name` (labels checkpoints and the
    train CLI's --algo), `algo` (the SB3 class name) and `defaults` (algorithm settings that suit
    small gridworlds better than SB3's Atari-oriented ones)."""

    name = "deep"
    algo = ""
    defaults: dict = {}

    def __init__(self, agent_id: str, seed: Optional[int] = None, policy: str = "mlp", device: str = "cpu",
                 **algo_kwargs):
        super().__init__(agent_id)
        if policy not in ("mlp", "cnn"):
            raise ValueError(f"policy must be 'mlp' or 'cnn', got {policy!r}")
        _sb3()
        self.seed = seed
        self.policy = policy
        self.device = device
        self.algo_kwargs = {**self.defaults, **algo_kwargs}
        self.model = None
        self.image: Optional[GridImage] = None

    # training
    def train(self, env: gymnasium.Env, steps: int = 50_000, seed: int = 0, out_dir=None,
              intent=None) -> list[dict]:
        """Train on `env` (a GridlockEnv) for `steps` environment steps and return one log row per
        finished episode: {episode, R, steps, terminated}, plus R_star when `intent` is given.

        `intent` only logs R* beside R and must come from outside gridlock.agents, which may not
        import the intent package. With `out_dir`, rows go to train_log.jsonl (the tabular
        learners' format, so report.plot_training_curves reads both) and the model is saved as
        `{name}_final.zip`."""
        out = Path(out_dir) if out_dir is not None else None
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
        with (open(out / "train_log.jsonl", "w") if out is not None else contextlib.nullcontext()) as log:
            logged = EpisodeLog(env, intent=intent, log=log)
            wrapped = GridImageObservation(logged)
            self.image = wrapped.image
            self.model = self._build_model(wrapped, seed)
            self.model.learn(total_timesteps=steps)
        if out is not None:
            self.save(out / f"{self.name}_final.zip")
        return logged.rows

    def _build_model(self, env: gymnasium.Env, seed: int):
        algorithm = getattr(_sb3(), self.algo)
        kwargs = dict(self.algo_kwargs)
        if self.policy == "cnn":
            from gridlock.agents.rl.networks import GridCNN
            policy_kwargs = dict(kwargs.pop("policy_kwargs", {}) or {})
            policy_kwargs.setdefault("features_extractor_class", GridCNN)
            policy_kwargs.setdefault("normalize_images", False)     # GridImage is already 0/1
            kwargs["policy_kwargs"] = policy_kwargs
            policy = "CnnPolicy"
        else:
            policy = "MlpPolicy"
        return algorithm(policy, env, seed=self.seed if self.seed is not None else seed, device=self.device,
                         verbose=0, **kwargs)

    # acting

    def reset(self, env) -> None:
        if self.image is None and hasattr(env, "scenario"):     # e.g. a model loaded from disk
            self.image = GridImage.for_scenario(env.scenario)

    def act(self, obs):
        if self.model is None or self.image is None:
            raise RuntimeError(f"{type(self).__name__} must be trained (or loaded and reset) before it acts")
        action, _ = self.model.predict(self.image(obs), deterministic=True)
        return ACTIONS[int(action)]

    # checkpoints

    def save(self, path) -> None:
        self.model.save(str(path))

    @classmethod
    def load(cls, path, agent_id: str, scenario=None, device: str = "cpu") -> "DeepAgent":
        """Restore a model written by save(). It unpickles code, so only load files you trust.
        Pass the scenario (or call reset with an env) before acting."""
        agent = cls(agent_id, device=device)
        agent.model = getattr(_sb3(), cls.algo).load(str(path), device=device)
        if scenario is not None:
            agent.image = GridImage.for_scenario(scenario)
        return agent


class DQNAgent(DeepAgent):
    """Off-policy, value-based: the deep counterpart of tabular Q-learning."""
    name, algo = "dqn", "DQN"
    defaults = {"learning_rate": 5e-4, "buffer_size": 50_000, "learning_starts": 1_000, "batch_size": 64,
                "gamma": 0.99, "train_freq": 4, "target_update_interval": 500, "exploration_fraction": 0.3,
                "exploration_final_eps": 0.05}


class PPOAgent(DeepAgent):
    """On-policy, policy-gradient with a clipped objective."""
    name, algo = "ppo", "PPO"
    defaults = {"n_steps": 1024, "batch_size": 64, "gamma": 0.99, "ent_coef": 0.01}


class A2CAgent(DeepAgent):
    """On-policy, synchronous advantage actor-critic."""
    name, algo = "a2c", "A2C"
    defaults = {"gamma": 0.99, "ent_coef": 0.01}


DEEP_ALGORITHMS = {cls.name: cls for cls in (DQNAgent, PPOAgent, A2CAgent)}


class EpisodeLog(gymnasium.Wrapper):
    """Records R (optionally R*) for each finished episode of a GridlockEnv, writing rows to
    `log` (an open file) as they finish. R* is read from the underlying State for analysis only."""

    def __init__(self, env: gymnasium.Env, intent=None, log=None):
        super().__init__(env)
        self.intent = intent
        self.log = log
        self.rows: list[dict] = []
        self._R = self._R_star = 0.0
        self._steps = 0

    def reset(self, **kwargs):
        self._R = self._R_star = 0.0
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._R += reward
        self._steps += 1
        if self.intent is not None:
            base = self.env.unwrapped
            self._R_star += self.intent.step_score(base.state, info["events"], base.learner_id)
        if terminated or truncated:
            row = {"episode": len(self.rows), "R": float(self._R), "steps": self._steps, "terminated": bool(terminated)}
            if self.intent is not None:
                row["R_star"] = float(self._R_star)
            self.rows.append(row)
            if self.log is not None:
                self.log.write(json.dumps(row) + "\n")
                self.log.flush()
        return obs, reward, terminated, truncated, info
