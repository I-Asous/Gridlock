"""Network pieces for the Stable-Baselines3 agents (imports torch --> gridlock.agents.rl.deep loads it lazily)."""

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class GridCNN(BaseFeaturesExtractor):
    """A small CNN for GridImage observations of any size: three 3x3 convolutions, then a linear
    layer. Unlike SB3's NatureCNN it works on boards as small as 5 x 5. On big boards (20+ cells a
    side) the last two convolutions use stride 2, so a 50 x 50 board doesn't need a huge linear layer."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        channels, height, width = observation_space.shape
        stride = 2 if min(height, width) >= 20 else 1
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=stride, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=stride, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, channels, height, width)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
