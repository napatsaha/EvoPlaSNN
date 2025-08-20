"""
2025-08-14
Simple T-Maze environment with cell number as state observation.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Literal, Tuple
from functools import partial


class TMaze(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete

    WALL = 0
    EMPTY = 1
    AGENT = 2
    REWARD = 3

    action_map = {
        0: np.array([-1, 0], dtype=np.int8),  # Up
        1: np.array([1, 0], dtype=np.int8),   # Down
        2: np.array([0, -1], dtype=np.int8),  # Left
        3: np.array([0, 1], dtype=np.int8)    # Right
    }

    action_names = {
        0: "Up",
        1: "Down",
        2: "Left",
        3: "Right"
    }

    def __init__(self, size=None, width=None, height=None, pad=1, *, 
                 max_steps=50, reward_function: Literal["A", "B"] = "A",
                 reward_min: float = 0.0, reward_max: float = 1.0):
        """
        Note on reward function:
        - 'A' is giving 1 when reaching target, 0 if truncated (max steps reached)
        - 'B' is giving a linearly-decreasing value (scaled between 0 and 1) 
            inversely proportional to the number of steps taken, ranging from min_steps (determined by size) to max_steps (user-determined).
        """
        super().__init__()
        self.width = size if width is None else width
        self.height = size if height is None else height
        self.pad = pad
        self.action_space = gym.spaces.Discrete(4)
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.reward_function = reward_function
        self.max_steps = int(max_steps)
        self._calculate_min_step()
        self._create_maze()
        self._create_reward_function()
        self.observation_space = gym.spaces.Discrete(self._num_state)
        self._step_count = 0

        # Set up plotting variables
        self.cmap = colors.ListedColormap([ "white", "black","blue", "green"])

    def _calculate_min_step(self):
        # Do this before padding
        up = self.height - 1
        right = (self.width + 1) // 2 - 1
        self.min_steps = int(up + right)

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
        self._starting_state = self._convert_pos_to_state(self._agent_pos)
        self._state_pos_dict = {state: self._convert_state_to_pos(state) for state in range(self._num_state)}

    def _create_reward_function(self):
        if self.reward_function == "A":
            def f(terminated, truncated, r, R, **kwargs):
                return R if terminated and not truncated else r
            self._reward_func = partial(f, r=self.reward_min, R=self.reward_max)
        elif self.reward_function == "B":
            self._reward_func = partial(self.f, m=self.min_steps, M=self.max_steps, r=self.reward_min, R=self.reward_max)
        else:
            raise ValueError(f"Invalid reward function: {self.reward_function}. Must be 'A' or 'B'.")

    def _reset_position(self):
        self.maze[np.unravel_index(self._empty_idx, self.maze.shape)] = self.EMPTY
        self.maze[*self._reward_pos] = self.REWARD
        self._agent_pos = self._convert_state_to_pos(self._starting_state)
        self.maze[*self._agent_pos] = self.AGENT
        self._step_count = 0

    @staticmethod
    def f(step, m, M, r, R, **kwargs):
        return (step - M) * (R - r) / (m - M) + r

    def render(self, fig_scale=1.0):
        fig, ax = plt.subplots(figsize=(self.width * fig_scale, self.height * fig_scale))
        ax.imshow(self.maze, cmap=self.cmap, extent=(0, self.width, 0, self.height), vmin=0, vmax=4, interpolation="nearest")
        for state, pos in self._state_pos_dict.items():
            ax.text(pos[1] + 0.5, self.height - pos[0] - 0.5, r"$s_{"+str(state)+r"}$", ha='center', va='center', fontsize=12, color=self.cmap(0))
        ax.set_xticks(np.arange(0, self.width, 1), labels=[])
        ax.set_yticks(np.arange(0, self.height, 1), labels=[])
        ax.grid(visible=True, color='gray', linewidth=1)
        plt.show()
        # return fig
    
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
        self._step_count = 0
        self._reset_position()
        state = self.get_agent_position()
        info = {
            'step_count': self._step_count,
            # 'reward_position': self.get_reward_position()
        }
        return state, info

    def step(self, action: int) -> Tuple[int, float | None, bool, bool, dict]:
        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        _, terminated = self._take_action(action)
        state = self.get_agent_position()
        info = {
            'step_count': self._step_count,
            # 'reward_position': self.get_reward_position()
        }
        # if truncated:
        #     reward = 0.0
        reward = self._reward_func(terminated=terminated, truncated=truncated, step=self._step_count)
        return state, reward, terminated, truncated, info

    def get_maze(self):
        return self.maze.copy()
    
    def get_agent_position(self):
        return self._convert_pos_to_state(self._agent_pos)
    
    def get_reward_position(self):
        return self._convert_pos_to_state(self._reward_pos)
    
    def _convert_pos_to_state(self, pos: np.ndarray):
        flat_idx = np.ravel_multi_index(pos, self.maze.shape)
        state_idx = np.isin(self._empty_idx, flat_idx).nonzero()[0]
        return state_idx.item()
    
    def _convert_state_to_pos(self, state: int):
        if state < 0 or state >= self._num_state:
            raise ValueError(f"State index out of bounds. Must be between [0, {self._num_state - 1}]. Got state: {state}")
        flat_idx = self._empty_idx[state]
        pos = np.unravel_index(flat_idx, self.maze.shape)
        return np.asarray(pos).reshape(2)

    def _take_action(self, action: int) -> Tuple[float | None, bool]:
        if action is None or action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action: {action}. Must be between [0, {self.action_space.n - 1}].")
        
        reward = None
        terminated = False

        # Perform movement
        new_pos = self._agent_pos + self.action_map[action]
        new_item = self.maze[*new_pos]

        # Check where the agent would end up
        if new_item == self.WALL:
            pass
        elif new_item == self.EMPTY:
            self.maze[tuple(self._agent_pos)] = self.EMPTY
            self._agent_pos = new_pos
            self.maze[tuple(self._agent_pos)] = self.AGENT
        elif new_item == self.REWARD:
            self.maze[tuple(self._agent_pos)] = self.EMPTY
            self._agent_pos = new_pos
            self.maze[tuple(self._agent_pos)] = self.AGENT
            reward = 1.0
            terminated = True
        
        return reward, terminated