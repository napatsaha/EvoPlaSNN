"""
2025-08-14
Simple T-Maze environment with cell number as state observation.
"""

from types import NoneType
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Literal, Tuple
from functools import partial


def man_dist(x, y):
    """A simplified Manhattan distance calculator"""
    return np.abs(x - y).sum()


class TMaze:
    pass

class BaseMaze(gym.Env):
    """
    Base maze class for any type of mazes.

    The only thing subclasses need to defined is the `_create_maze()` method, which will fill up initial maze array,
    with corresponding values for walls, empty cells, agent starting position, good and bad cells.

    The maze is represented as a 2D numpy array with the following values:
        0: Wall
        1: Empty cell
        2: Agent
        3: Good (reward) cell
        4: Bad (punishment) cell

    Attributes:
        width (int): The width of the environment.
        height (int): The height of the environment.
        pad (int): The padding around the environment.
        reward_bad (float): The reward for bad actions.
        reward_good (float): The reward for good actions.
        reward_trunc (float): The reward for truncation or termination.
        penalty (float): The penalty for invalid actions or other undesirable behavior.
        reward_inter (float): The intermediate reward for certain actions.
        max_steps (int): The maximum number of steps allowed in the environment.
        observation_space (gym.spaces.Discrete): The observation space of the environment.
        action_space (gym.spaces.Discrete): The action space of the environment.
        _step_count (int): The current step count in the environment.
        cmap (colors.ListedColormap): The colormap used for plotting the environment.
    """
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete

    WALL = 0
    EMPTY = 1
    AGENT = 2
    GOOD = 3
    BAD = 4

    COLOR_LIST = ["white", "black", "blue", "green", "red"]

    action_map = {
        -1: np.array([0, 0], dtype=np.int8), # Stationary
        0: np.array([-1, 0], dtype=np.int8),  # Up
        1: np.array([1, 0], dtype=np.int8),   # Down
        2: np.array([0, -1], dtype=np.int8),  # Left
        3: np.array([0, 1], dtype=np.int8)    # Right
    }

    action_names = {
        -1: "Stay",
        0: "Up",
        1: "Down",
        2: "Left",
        3: "Right"
    }

    def __init__(self, size=None, width=None, height=None, pad=1, *, 
                 max_steps=50, randomise_start: bool = False, random_min_dist: int = 0,
                 obs_type: Literal["state", "position", "surroundings"] = "state",
                 include_agent_pos: bool = False,
                 reward_step_closer: bool = False,
                 terminate_on_crash: bool = False,
                 penalty: float = -0.1, reward_inter: float = 0.1, reward_null: float | NoneType = 0.0,
                 reward_bad: float = -1.0, reward_good: float = 1.0, reward_trunc: float = -1.0,
                 **kwargs):
        """
        Initializes the environment with the given parameters.

        Args:
            size (int, optional): The size of the environment. If `width` and `height` 
                are not provided, this value is used for both dimensions. Defaults to None.

            width (int, optional): The width of the environment. Defaults to None.

            height (int, optional): The height of the environment. Defaults to None.

            pad (int, optional): The outer padding of Walls around the maze. Defaults to 1.

            max_steps (int, optional): The maximum number of environment steps before truncation. Defaults to 50.

            randomise_start (bool, optional): Whether or not to allow agent to spawn at random position at beginning of each episode. Defaults to False.

            random_min_dist (int, optional): If `randomise_start=True`, this variable controls which cells are allowed for random agent spawn position. If
                the Euclidean distance between a cell and either of the good or bad reward is greater than or equal to `random_min_dist`, then that cell is included.
                Defaults to 0.

            obs_type (str, optional): Determines what type of observation to be returned:
                - `state` returns a single integer scalar denoting the cell index of the maze the agent is in.
                - `position` returns a i-j indexing tuple of the position of the agent
                - `surrounding` returns 8-length array showing the type of objects (0-4) in the 8 adjacent cells of the agent.
                Defaults to `state`.

            reward_step_closer (bool, optional): Whether to reward the agent for stepping closer to the goal 
                (by comparing whether the Manhattan distance between agent and goal is closer than has ever been in this episode). Defaults to False.

            terminate_on_crash (bool, optional): Controls whether the episode will terminate when agent moves into a Wall cell. If True, the end-of-episode reward
                will use the `penalty` parameter. Defaults to False.

            penalty (float, optional): The returned reward when an agent bumps into a Wall cell. Defaults to -0.1.

            reward_inter (float, optional): The intermediate reward for either stepping into empty cell (if `reward_step_closer=False`)
                or getting closer to goal (if `reward_step_closer=True`). Defaults to 0.1.

            reward_null (float, NoneType, optional): The intermediate reward which covers situations not covered by other reward types.
                Defaults to 0.        
        
            reward_bad (float, optional): The terminal reward for ending up in the bad final state. Defaults to -1.0.

            reward_good (float, optional): The terminal reward for ending up in the good final state. Defaults to 1.0.

            reward_trunc (float, optional): The terminal reward for reaching maximum steps before reaching good or bad states. Defaults to -1.0.
    
        """
        super().__init__()
        self.width = size if width is None else width
        self.height = size if height is None else height
        self.area = self.width * self.height if self.width is not None and self.height is not None else None
        self.pad = pad
        self.max_steps = int(max_steps)

        self.randomise_start = randomise_start
        self._random_min_dist = max(0, int(random_min_dist))

        # Reward parameters
        # Final episode reward
        self.reward_bad = reward_bad
        self.reward_good = reward_good
        self.reward_trunc = reward_trunc
        # Intermediate reward
        self.penalty = penalty
        self.reward_inter = reward_inter
        self.reward_null = reward_null
        # Controls for reward function
        self._check_closest_distance = reward_step_closer
        self.terminate_on_crash = terminate_on_crash

        # Observation type
        if obs_type not in ["state", "position", "surroundings"]:
            raise ValueError(f"Invalid obs_type: {obs_type}. Must be 'state', 'position' or 'surroundings'.")
        self.obs_type = obs_type
        self._use_obs_state = obs_type == "state"
        self._use_obs_pos = obs_type == "position"
        self._use_obs_surr = obs_type == "surroundings"
        if self._use_obs_surr and include_agent_pos:
            self.neighbouring = np.array([[-1, -1], [-1, 0], [-1, 1],
                                          [0, -1],  [0, 0], [0, 1],
                                          [1, -1], [1, 0], [1, 1]])
        elif self._use_obs_surr and not include_agent_pos:
            self.neighbouring = np.array([[-1, -1], [-1, 0], [-1, 1],
                                          [0, -1],           [0, 1],
                                          [1, -1], [1, 0], [1, 1]])

        # Create important attributes
        self._create_maze()
        self._prepare_positions()
        # self._create_reward_function()
        self._calculate_min_step()

        # Gym spaces attributes
        if self._use_obs_state:
            self.observation_space = gym.spaces.Discrete(self._num_state)
        elif self._use_obs_pos:
            self.observation_space = gym.spaces.Box(low=np.zeros((2,), dtype=np.int8) + self.pad,
                                                    high=np.array((self.height, self.width)) - self.pad - 1,
                                                    shape=(2,), dtype=np.int8)
        elif self._use_obs_surr:
            self.observation_space = gym.spaces.Box(low=0, high=4, shape=(self.neighbouring.shape[0],), dtype=np.int8)
        else:
            raise ValueError("Invalid observation type.")
        self.action_space = gym.spaces.Discrete(4)
        self._step_count = 0
        self._closest_dist = self.min_steps
        self._prev_dist = None

        # Plotting attributes
        self.cmap = colors.ListedColormap(self.COLOR_LIST)

    ## Creation functions
    def _calculate_min_step(self):
        # # Do this before padding
        # up = self.height - 1
        # right = (self.width + 1) // 2 - 1
        # self.min_steps = int(up + right)
        self.min_steps = man_dist(self._agent_pos, self._good_pos).item()

    def _create_maze(self):
        raise NotImplementedError("Subclasses must implement the `_create_maze()` method.")
        # # Create empty array (filled with walls, as 1's)
        # self.maze = np.full((self.height, self.width), dtype=np.int8, fill_value=self.WALL)
        # # Add traversing paths (as 0's)
        # mid_width = self.width // 2
        # self.maze[:, mid_width] = self.EMPTY
        # self.maze[0, :] = self.EMPTY
        # # Set agent starting position
        # self.maze[-1, mid_width] = self.AGENT
        # # Set good position at right wing of T
        # self.maze[0, -1] = self.GOOD
        # # Set bad position at left wing of T
        # self.maze[0, 0] = self.BAD

    def _prepare_positions(self):
        # Add walls
        self.maze = np.pad(self.maze, self.pad, mode='constant', constant_values= self.WALL)
        self.width += 2 * self.pad
        self.height += 2 * self.pad
        # Update position and state values
        self._num_state = np.count_nonzero(self.maze)
        self._empty_idx = np.flatnonzero(self.maze)
        self._agent_pos = np.argwhere(self.maze == self.AGENT)[0]
        self._good_pos = np.argwhere(self.maze == self.GOOD)[0]
        self._bad_pos = np.argwhere(self.maze == self.BAD)[0]
        if self.randomise_start:
            # Pre-determine list of indices which is at least `random_min_dist` away from either good or bad rewards
            self._idx_dist_rec = []
            for idx in self._empty_idx:
                idx2d = np.unravel_index(idx, self.maze.shape)
                good_dist = man_dist(self._good_pos, idx2d)
                bad_dist = man_dist(self._bad_pos, idx2d)
                self._idx_dist_rec.append((int(idx), idx2d, int(good_dist), int(bad_dist)))
            self._valid_idx = [idx for idx, _, gd, bd in self._idx_dist_rec if gd >= self._random_min_dist and bd >= self._random_min_dist]
            # Randomly choose starting position from this list
            self.maze[*self._agent_pos] = self.EMPTY
            rand_idx = self.np_random.choice(self._valid_idx)
            self._agent_pos = np.unravel_index(rand_idx, self.maze.shape)
            self.maze[*self._agent_pos] = self.AGENT
        else:
            self._starting_pos = self._agent_pos.copy()
        self._state_pos_dict = {state: self._convert_state_to_pos(state) for state in range(self._num_state)}

    ## Internal functions
    def _reset_position(self):
        self.maze[np.unravel_index(self._empty_idx, self.maze.shape)] = self.EMPTY
        self.maze[*self._good_pos] = self.GOOD
        self.maze[*self._bad_pos] = self.BAD
        if self.randomise_start:
            rand_idx = self.np_random.choice(self._valid_idx)
            self._agent_pos = np.unravel_index(rand_idx, self.maze.shape)
        else:
            self._agent_pos = self._starting_pos.copy()
        self.maze[*self._agent_pos] = self.AGENT
        # self._step_count = 0

    def _take_action(self, action: int, info: dict) -> Tuple[float, dict]:
        """
        Internals of a step function:
        - Apply chosen action to current agent position to see what item would be displaced
        - If it's a wall, cancel the motion. 
        - Otherwise, calculate and apply the new position to agent
        - Calculate the Manhattan distance and update closest distant if required
        - Finally calculate reward and whether the episode is now terminated

        The info dictionary are there to store intermediate information
        """
        terminated = False
        reward = 0.0
        new_dist = self._prev_dist

        # Perform supposed movement
        new_pos = self._agent_pos + self.action_map[action]
        displaced_item = self.maze[*new_pos]

        # Check where the agent would end up
        if displaced_item == self.WALL:
            # Bumping into wall
            reward = self.penalty
            if self.terminate_on_crash:
                terminated = True
        elif displaced_item == self.AGENT:
            # Stationary case
            reward = self.reward_null
        else:
            # Valid movement possible
            self.maze[tuple(self._agent_pos)] = self.EMPTY
            self._agent_pos = new_pos
            self.maze[tuple(self._agent_pos)] = self.AGENT

            # Calculate new closest distance
            if self._check_closest_distance:
                new_dist = man_dist(self._agent_pos, self._good_pos).item()
                self._prev_dist = new_dist
                if new_dist < self._closest_dist:
                    self._closest_dist = new_dist
                    reward = self.reward_inter
                else:
                    reward = self.reward_null
            else:
                reward = self.reward_null

            # Calculate reward and terminate for non-empty cells
            if displaced_item == self.GOOD:
                reward = self.reward_good
                terminated = True
            elif displaced_item == self.BAD:
                reward = self.reward_bad
                terminated = True
            else:
                pass

        info['manhattan_dist'] = new_dist
        info["closest_dist"] = self._closest_dist
        info["terminated"] = terminated

        return reward, info
    
    ## Helper functions
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

    ## Gym functions
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
        state = self.get_observation()
        info = {
            'step_count': self._step_count,
            # 'reward_position': self.get_reward_position()
        }
        if self._check_closest_distance:
            self._calculate_min_step()
            self._closest_dist = self.min_steps
            self._prev_dist = self._closest_dist
            info['closest_dist'] = self._closest_dist
        return state, info

    def step(self, action: int) -> Tuple[int, float | None, bool, bool, dict]:
        self._step_count += 1
        info = {'step_count': self._step_count}
        truncated = self._step_count >= self.max_steps
        reward, info = self._take_action(action, info)
        terminated = info.get('terminated', None)
        if truncated and not terminated:
            reward = self.reward_trunc
        state = self.get_observation()
        return state, reward, terminated, truncated, info

    ## Other functions
    @staticmethod
    def f(step, m, M, r, R, **kwargs):
        return (step - M) * (R - r) / (m - M) + r

    def get_maze(self):
        return self.maze.copy()
    
    def get_observation(self):
        if self._use_obs_state:
            return self._convert_pos_to_state(self._agent_pos)
        elif self._use_obs_pos:
            return np.asarray(self._agent_pos)
        elif self._use_obs_surr:
            indices = self._agent_pos + self.neighbouring
            obs = self.maze[indices[:, 0], indices[:, 1]]
            return obs
    
    def get_good_state(self):
        return self._convert_pos_to_state(self._good_pos)
    
    def get_bad_state(self):
        return self._convert_pos_to_state(self._bad_pos)
    
    def get_agent_state(self):
        return self._convert_pos_to_state(self._agent_pos)
    
    def get_agent_position(self):
        return np.array(self._agent_pos)

    def get_min_reward(self):
        return min(self.reward_bad, self.reward_good, self.penalty, self.reward_trunc, self.reward_inter)
    
    def get_max_reward(self):
        return max(self.reward_bad, self.reward_good, self.penalty, self.reward_trunc, self.reward_inter)

    @property
    def reward_list(self):
        return sorted(set([0.0, self.reward_bad, self.reward_good, self.penalty, self.reward_trunc, self.reward_inter]))

    @property
    def random_min_dist(self):
        return self._random_min_dist
    
    @random_min_dist.setter
    def random_min_dist(self, value: int):
        self._random_min_dist = max(0, int(value))
        # Updates list of valid indices a set distance from either goals
        self._valid_idx = [idx for idx, _, gd, bd in self._idx_dist_rec if gd >= self._random_min_dist and bd >= self._random_min_dist]

    # def _create_reward_function(self):
    #     if self.reward_function == "A":
    #         def f(terminated, truncated, reward, **kwargs):
    #             if terminated:
    #                 return reward
    #             elif truncated:
    #                 return self.reward_trunc
    #             else:
    #                 return reward
    #         self._reward_func = partial(f)
    #     elif self.reward_function == "B":
    #         self._reward_func = partial(self.f, m=self.min_steps, M=self.max_steps, r=0.0, R=1.0)
    #     elif self.reward_function == "C":
    #         def f(reward, **kwargs):
    #             return reward
    #         self._reward_func = partial(f)
    #     else:
    #         raise ValueError(f"Invalid reward function: {self.reward_function}. Must be 'A' or 'B'.")

    # def _reward_func(self, displaced_item: int, new_dist: int | None) -> Tuple[float, bool]:
    #     """
    #     Return a tuple of (reward, terminated) as a function of the displaced item from the chosen action.
    #     """
    #     terminated = False

    #     if displaced_item == self.WALL:
    #         reward = self.penalty
    #     elif displaced_item == self.EMPTY:
    #         if self._reward_step_closer:
    #             if new_dist < self._closest_dist:
    #                 self._closest_dist = new_dist
    #                 reward = self.reward_inter
    #             else:
    #                 reward = 0.0
    #         else:
    #             reward = self.reward_inter
    #     elif displaced_item == self.GOOD:
    #         reward = self.reward_good
    #         terminated = True
    #     elif displaced_item == self.BAD:
    #         reward = self.reward_bad
    #         terminated = True
    #     elif displaced_item == self.AGENT:
    #         # Stationary case
    #         reward = 0.0
    #     else:
    #         raise ValueError(f"Invalid displaced item: {displaced_item}.")

    #     return reward, terminated

    

