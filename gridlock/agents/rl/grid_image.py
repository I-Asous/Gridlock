"""Grid-image observations for neural-network learners.

The tabular learners key a Q-table on the observation vector (positions and flag bits, see
gridlock.env.observation). A network learns better from a picture of the board: one channel per
kind of thing, each an H x W map of 0s and 1s, so a convolution or an MLP can share what it
learns across cells. Channels, in order:

    0 wall   1 goal   2 checkpoint   3 trap   4 switch      (the level's tiles, fixed)
    5 self   6 other agents   7 objects                      (from the observation vector)
    8.. one per state flag, all 1s while that flag is on

The image is built from the same observation vector the tabular learners see, plus the static
tile layout, so a deep learner observes exactly what they do. Like them it never sees R*, and
it can't tell first visits to checkpoints from repeats.
"""

import gymnasium
import numpy as np
from gymnasium import spaces

from gridlock.env.grid import Tile
from gridlock.env.observation import ObservationEncoder

TILE_CHANNELS = (Tile.WALL, Tile.GOAL, Tile.CHECKPOINT, Tile.TRAP, Tile.SWITCH)
SELF, OTHERS, OBJECTS = 5, 6, 7


class GridImage:
    """Turns observation vectors from `encoder` into (channels, height, width) float32 images."""

    def __init__(self, grid, encoder: ObservationEncoder):
        self.width, self.height = grid.width, grid.height
        self.n_agents = len(encoder.agent_ids)
        self.n_objects = len(encoder.object_ids)
        self.n_flags = len(encoder.flag_names)
        self.channels = OBJECTS + 1 + self.n_flags
        self.static = np.zeros((len(TILE_CHANNELS), self.height, self.width), dtype=np.float32)
        for y, row in enumerate(grid.tiles):
            for x, tile in enumerate(row):
                if tile in TILE_CHANNELS:
                    self.static[TILE_CHANNELS.index(tile), y, x] = 1.0
        self.space = spaces.Box(0.0, 1.0, (self.channels, self.height, self.width), dtype=np.float32)

    @classmethod
    def for_scenario(cls, scenario) -> "GridImage":
        return cls(scenario.grid, ObservationEncoder.for_scenario(scenario))

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs).ravel()
        image = np.zeros(self.space.shape, dtype=np.float32)
        image[:len(TILE_CHANNELS)] = self.static
        positions = obs[:2 * (self.n_agents + self.n_objects)].reshape(-1, 2)
        for k, (x, y) in enumerate(positions):
            if x >= self.width or y >= self.height:          # (width, height) means "not on the board"
                continue
            channel = SELF if k == 0 else OTHERS if k < self.n_agents else OBJECTS
            image[channel, y, x] = 1.0
        for k, bit in enumerate(obs[2 * (self.n_agents + self.n_objects):]):
            if bit:
                image[OBJECTS + 1 + k] = 1.0
        return image


class GridImageObservation(gymnasium.ObservationWrapper):
    """Wraps a GridlockEnv (or anything whose `unwrapped` has `world.scenario` and `encoder`) so
    observations are GridImage images instead of position vectors."""

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        base = env.unwrapped
        self.image = GridImage(base.world.scenario.grid, base.encoder)
        self.observation_space = self.image.space

    def observation(self, observation):
        return self.image(observation)
