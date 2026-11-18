from gridlock.agents.rl.q_learning import QLearningAgent
from gridlock.agents.rl.sarsa import SARSAAgent

ALGORITHMS = {cls.name: cls for cls in (QLearningAgent, SARSAAgent)}   # --algo name -> agent class

__all__ = ["ALGORITHMS", "QLearningAgent", "SARSAAgent"]
