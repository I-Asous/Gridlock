"""Tabular SARSA: the on-policy counterpart of QLearningAgent. Trained on R only.

The only change from Q-learning is the bootstrap target. Q-learning backs up max_a Q(s', a);
SARSA backs up Q(s', a'), where a' is the action the ε-greedy policy actually takes next. SARSA
therefore learns the value of the policy it is following, exploration included, and tends to
prefer routes that stay safe under random missteps.
"""

from gridlock.agents.rl.q_learning import QLearningAgent, action_index, state_key


class SARSAAgent(QLearningAgent):
    name = "sarsa"
    # the training loop chooses a' before calling update()
    on_policy = True

    def update(self, obs, action, reward: float, next_obs, terminated: bool, next_action=None) -> None:
        """One SARSA backup. `next_action` is the action chosen in next_obs. It is required unless
        `terminated`. As with Q-learning, a time-limit truncation still bootstraps."""
        row = self.q[state_key(obs)]
        if terminated:
            future = 0.0
        elif next_action is None:
            raise ValueError("SARSA needs next_action unless the episode terminated")
        else:
            next_row = self.q.get(state_key(next_obs))
            future = 0.0 if next_row is None else float(next_row[action_index(next_action)])
        i = action_index(action)
        row[i] += self.alpha * (reward + self.gamma * future - row[i])
