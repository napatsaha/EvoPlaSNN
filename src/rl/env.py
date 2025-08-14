"""
2025-08-14
Simple T-Maze environment with cell number as state observation.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class TMaze(gym.Env):
    WALL = 0
    EMPTY = 1
    AGENT = 2
    REWARD = 3

    def __init__(self, size=None, width=None, height=None, pad=1):
        super().__init__()
        self.width = size if width is None else width
        self.height = size if height is None else height
        self.pad = pad
        self.action_space = gym.spaces.Discrete(4)
        self._create_maze()
        self.observation_space = gym.spaces.Discrete(self._num_state)

        # Set up plotting variables and Enums
        self.cmap = colors.ListedColormap([ "white", "black","blue", "green"])


    def _create_maze(self):
        # Create empty array (filled with walls, as 1's)
        self.maze = np.full((self.height, self.width), dtype=np.int8, fill_value=self.WALL)
        # Add traversing paths (as 0's)
        mid_width = self.width // 2
        self.maze[:, mid_width] = self.EMPTY
        self.maze[0, :] = self.EMPTY
        # Set agent starting position
        self.maze[-1, mid_width] = self.AGENT
        # Set reward position at right wing of T
        self.maze[0, -1] = self.REWARD
        # Add walls
        self.maze = np.pad(self.maze, self.pad, mode='constant', constant_values= self.WALL)
        self.width += 2 * self.pad
        self.height += 2 * self.pad
        # Update position and state values
        self._num_state = np.count_nonzero(self.maze)
        self._empty_idx = np.flatnonzero(self.maze)
        self._agent_pos = np.argwhere(self.maze == self.AGENT)[0]
        self._reward_pos = np.argwhere(self.maze == self.REWARD)[0]
        self._state_pos_dict = {state: self._convert_state_to_pos(state) for state in range(self._num_state)}

    def render(self, fig_scale=1.0):
        fig, ax = plt.subplots(figsize=(self.width * fig_scale, self.height * fig_scale))
        ax.imshow(self.maze, cmap=self.cmap, extent=(0, self.width, 0, self.height), vmin=0, vmax=4, interpolation="nearest")
        for state, pos in self._state_pos_dict.items():
            ax.text(pos[1] + 0.5, self.height - pos[0] - 0.5, r"$s_{"+str(state)+r"}$", ha='center', va='center', fontsize=12, color=self.cmap(0))
        ax.set_xticks(np.arange(0, self.width, 1), labels=[])
        ax.set_yticks(np.arange(0, self.height, 1), labels=[])
        ax.grid(visible=True, color='gray', linewidth=1)
        plt.show()

    def get_maze(self):
        return self.maze.copy()
    
    def get_agent_position(self):
        return self._convert_pos_to_state(self._agent_pos)
    
    def get_reward_position(self):
        return self._convert_pos_to_state(self._reward_pos)
    
    def _convert_pos_to_state(self, pos: np.ndarray):
        flat_idx = np.ravel_multi_index(pos, self.maze.shape)
        state_idx = np.isin(self._empty_idx, flat_idx).nonzero()[0]
        return state_idx
    
    def _convert_state_to_pos(self, state: int):
        if state < 0 or state >= self._num_state:
            raise ValueError(f"State index out of bounds. Must be between [0, {self._num_state - 1}]. Got state: {state}")
        flat_idx = self._empty_idx[state]
        pos = np.unravel_index(flat_idx, self.maze.shape)
        return np.asarray(pos)

