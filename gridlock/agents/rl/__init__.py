from gridlock.agents.rl.deep import DEEP_ALGORITHMS, A2CAgent, DQNAgent, PPOAgent
from gridlock.agents.rl.q_learning import QLearningAgent
from gridlock.agents.rl.sarsa import SARSAAgent

ALGORITHMS = {cls.name: cls for cls in (QLearningAgent, SARSAAgent)}   # tabular: --algo name -> agent class

__all__ = ["A2CAgent", "ALGORITHMS", "DEEP_ALGORITHMS", "DQNAgent", "PPOAgent", "QLearningAgent", "SARSAAgent"]
