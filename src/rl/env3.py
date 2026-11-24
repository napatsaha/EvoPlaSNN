"""
2026-11-17

Another alternate version based on `env2.BaseGridMaze` 

Key differences are:
- Agent have a direction it is facing
"""

from types import NoneType
from typing import List, Literal, Tuple, Dict
from functools import partial
import warnings
from enum import IntEnum

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patches

from common.base import SeasonalEnvProtocol


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Rotation(IntEnum):
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1

class ObjType(IntEnum):
    EMPTY = 0
    WALL = 1
    AGENT = 2
    GREEN = 3
    BLUE = 4

OBJ_COLOR = ["white", "black", "red", "green", "blue"]


ACTION_MAP = {
    0: Rotation.LEFT,
    1: Rotation.STRAIGHT,
    2: Rotation.RIGHT
}

ACTION_NAMES = {
    0: "Turn Left",
    1: "Straight",
    2: "Turn Right"
}


def rotate(angle: int | float) -> Tuple[int | int]:
    """
    Find the (i,j) coordinates for a clockwise angle starting from upwards direction.

    Can be used to find final transition from Direction + Rotation

    Args:
        angle (int | float): Clockwise angle from upwards position, in units of $\pi/2$ or 90 degrees. If negative will be anti-clockwise.

    Returns:
        Tuple[int | int]: An (i,j) coordinate translation

    ---
    Example:

    Find the cell position of a certain rotation away from an agent facing a certain direction
    ```
    >>> import numpy as np

    >>> agent_pos = np.array([3,4])
    >>> agent_direction = Direction.DOWN

    # Find the cell to the right of the direction the agent is facing
    >>> rotate(agent_direction + Rotation.RIGHT) + agent_pos

    array([3,3])
    ```
    """
    i = -np.cos(angle*np.pi/2).round().astype(int)
    j = np.sin(angle*np.pi/2).round().astype(int)
    return i, j


class BaseOpenMaze(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.MultiDiscrete

    _base_reward_function = dict(
        desired_eat = 1.0,
        desired_avoid = -1.0,
        undesired_eat = -1.0,
        undesired_avoid = 0.0,
        wall_eat = -1.0,
        wall_avoid = 1.0,
        wall_hug = -1.0,
        empty_straight = 1.0,
        empty_turn = -1.0,
    )

    def __init__(self, size=None, width=None, height=None, *, 
                 max_steps: int = None, 
                #  randomise_start: bool = False, random_spawn_method: Literal["random", "area"] = "random",
                #  random_spawn_area: List[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                #  obs_type: Literal["position", "surroundings", "sensors"] = "sensors",
                 num_green: int = 1, num_blue: int = 1,
                 desired_obj: Literal["green", "blue"] = "green",
                 undesired_obj: Literal["green", "blue"] = "blue",
                #  include_agent_pos: bool = False,
                #  terminate_on_crash: bool = False,
                #  penalty: float = -0.1, reward_inter: float = 0.1, reward_null: float | NoneType = 0.0,
                #  reward_bad: float = -1.0, reward_good: float = 1.0, reward_trunc: float = 0.0,
                 reward_func_params: Dict[str, float | int] = None,
                 **kwargs):
        """
        Initializes the environment with the given parameters.

        Args:
            size (int, optional): The size of the environment. If `width` and `height` 
                are not provided, this value is used for both dimensions. Defaults to None.

            width (int, optional): The width of the environment. Defaults to None.

            height (int, optional): The height of the environment. Defaults to None.

            max_steps (int, optional): The maximum number of environment steps before truncation. Defaults to 50.

            num_green (int, optional): Number of GREEN-type objects to be put into the maze. Defaults to 1.

            num_blue (int, optional): Number of BLUE-type objects to be put into the maze. Defaults to 1.

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
        # self.pad = 1
        self.max_steps = int(max_steps) if max_steps is not None else np.inf

        # Randomising starting position
        # self.randomise_start = randomise_start

        # Reward parameters
        # Final episode reward
        self.reward_function = self._base_reward_function
        if reward_func_params is not None:
            self.reward_function.update(reward_func_params)
        # self.reward_bad = reward_bad
        # self.reward_good = reward_good
        # self.reward_trunc = reward_trunc
        # Intermediate reward
        # self.penalty = penalty
        # self.reward_inter = reward_inter
        # self.reward_null = reward_null
        # Controls for reward function
        # self.terminate_on_crash = terminate_on_crash

        # Observation type
        self._sensor_direction = (Rotation.LEFT, Rotation.STRAIGHT, Rotation.RIGHT)

        # Food settings
        self._num_green = num_green
        self._num_blue = num_blue
        # Initialise which object is the desired/good object
        if desired_obj == "green":
            self._desired_objtype = ObjType.GREEN
        elif desired_obj == "blue":
            self._desired_objtype = ObjType.BLUE
        else:
            raise ValueError(f"Invalid 'desired_obj'={desired_obj}")
        # Same for undesired/bad object
        if undesired_obj == "green":
            self._undesired_objtype = ObjType.GREEN
        elif undesired_obj == "blue":
            self._undesired_objtype = ObjType.BLUE
        else:
            raise ValueError(f"Invalid 'undesired_obj'={undesired_obj}")
        assert self._desired_objtype != self._undesired_objtype, "Desired and Undesired objects must not be the same. Got " + \
            f"desired object: {self._desired_objtype}, and undesired object: {self._undesired_objtype}"

        # Agent descriptors
        self._agent_pos: np.ndarray = None
        self._agent_direction: Direction = None

        # Create maze array
        self._create_maze()
        self._initialise_positions()
        self._reset_position()

        # Gym spaces attributes
        
        self.observation_space = gym.spaces.MultiDiscrete([len(ObjType)-1 for _ in range(len(self._sensor_direction))], dtype=np.int8)

        self.action_space = gym.spaces.Discrete(len(Rotation))
        
        # Counters
        self._step_count = 0
        self._blue_count = 0
        self._green_count = 0

        # Plotting attributes
        self.cmap = colors.ListedColormap(OBJ_COLOR)

    ## Creation functions

    def _create_maze(self):
        """Initialise maze + padding"""
        self.maze: np.ndarray = np.full((self.height, self.width), dtype=np.int8, fill_value=ObjType.EMPTY)
        self._original_shape = self.maze.shape

        # Add walls
        # self._num_state = np.prod(self._original_shape)
        self.maze = np.pad(self.maze, 1, mode='constant', constant_values= ObjType.WALL)
        self.width += 2 #* self.pad
        self.height += 2 #* self.pad

    def _initialise_positions(self):
        """Set up agent and food positions"""
        available_flat_idx = np.arange(np.prod(self._original_shape))
        np.random.shuffle(available_flat_idx)
        # Get non-overlapping flat indices
        _blue_idx = available_flat_idx[:self._num_blue]
        _green_idx = available_flat_idx[self._num_blue:(self._num_blue+self._num_green)]

        # Turn flat indices into (i,j) coordinates within padded maze
        _blue_pos = np.c_[np.unravel_index(_blue_idx, self._original_shape)] + 1
        _green_pos = np.c_[np.unravel_index(_green_idx, self._original_shape)] + 1

        # Apply food objects to randomised positions
        for pos in _blue_pos:
            self.maze[*pos] = ObjType.BLUE
        for pos in _green_pos:
            self.maze[*pos] = ObjType.GREEN

        # Find and allocate unoccupied position for agent
        self._agent_pos = self._find_unoccupied_position()
        self._agent_direction = np.random.randint(len(Direction))
        self.maze[*self._agent_pos] = ObjType.AGENT

    def _find_unoccupied_position(self) -> np.ndarray:
        """Sample maze positions until an empty cell is found"""
        while True:
            i = np.random.randint(1, self.height-1)
            j = np.random.randint(1, self.width-1)
            if self.maze[i, j] == ObjType.EMPTY:
                return np.array([i,j])

    ## Internal functions
    def _reset_position(self):
        """
        Reset all positions, and randomise agent position if necessary.
        """
        # Reset all non-wall positions to empty
        self.maze[1:(self.height-1), 1:(self.width-1)] = ObjType.EMPTY
        # Reinitialise food and agent positiobs
        self._initialise_positions()
        

    def _get_observation(self) -> np.ndarray:
        """Take item type for each sensor direction from the agent position"""
        obs = []
        for sensor in self._sensor_direction:
            cell_pos = self._agent_pos + rotate(self._agent_direction + sensor)
            item_type = self.maze[*cell_pos]
            obs.append(item_type)
        return np.array(obs)

    def _get_reward(self, obs, action) -> float | int:
        """Calculate reward from current observation and supposed action"""
        item_displaced = obs[action]

        if self._desired_objtype in obs:
            if item_displaced == self._desired_objtype:
                reward = self.reward_function["desired_eat"]
            else:
                reward = self.reward_function["desired_avoid"]
        elif self._undesired_objtype in obs:
            if item_displaced == self._undesired_objtype:
                reward = self.reward_function["undesired_eat"]
            else:
                reward = self.reward_function["undesired_avoid"]
        elif ObjType.WALL in obs:
            if obs[Rotation.STRAIGHT+1] == ObjType.WALL:
                if (action-1) == Rotation.STRAIGHT:
                    reward = self.reward_function["wall_eat"]
                else:
                    reward = self.reward_function["wall_avoid"]
            elif obs[Rotation.LEFT+1] == ObjType.WALL:
                if (action-1) == Rotation.RIGHT:
                    reward = self.reward_function["wall_avoid"]
                elif (action-1) == Rotation.STRAIGHT:
                    reward = self.reward_function["wall_hug"]
                else:
                    reward = self.reward_function["wall_eat"]
            elif obs[Rotation.RIGHT+1] == ObjType.WALL:
                if (action-1) == Rotation.LEFT:
                    reward = self.reward_function["wall_avoid"]
                elif (action-1) == Rotation.STRAIGHT:
                    reward = self.reward_function["wall_hug"]
                else:
                    reward = self.reward_function["wall_eat"]
        elif ObjType.EMPTY in obs:
            if (action-1) == Rotation.STRAIGHT:
                reward = self.reward_function["empty_straight"]
            else:
                reward = self.reward_function["empty_turn"]
        else:
            reward = 0.0

        return reward

    def _take_action(self, action: int) -> Tuple[float, dict]:
        """
        Internals of a step function:
        - Apply chosen action to current agent position to see what item would be displaced
        - If it's a wall, cancel the motion, but change the direction. 
        - Otherwise, calculate and apply the new position to agent
        - Finally calculate reward 

        The info dictionary are there to store intermediate information
        """
        # reward = 0.0

        # TODO: Validate action is within ACTION_MAP

        # Perform supposed movement
        new_pos = self._agent_pos + rotate(self._agent_direction + ACTION_MAP[action])
        displaced_item = self.maze[*new_pos]

        # Check where the agent would end up
        if displaced_item == ObjType.WALL:
            # Bumping into wall
            self._agent_direction = (self._agent_direction + ACTION_MAP[action]) % 4
        # elif displaced_item == ObjType.AGENT:
        #     # Stationary case
        else:
            if (displaced_item == ObjType.BLUE) or (displaced_item == ObjType.GREEN):
                # If displaced object is a food object, find new position before last one is replaced
                # (to prevent low possibility of respawning in previous agent location)
                pos = self._find_unoccupied_position()
            # Perform movement by replacing new pos with agent, and removing agent from previous pos
            self.maze[*self._agent_pos] = ObjType.EMPTY
            self.maze[*new_pos] = ObjType.AGENT
            # Update agent position and direction
            self._agent_pos = new_pos
            self._agent_direction = (self._agent_direction + ACTION_MAP[action]) % 4


            # Calculate reward and terminate for non-empty cells
            if displaced_item == ObjType.GREEN:
                # Respawn food object
                self.maze[*pos] = ObjType.GREEN

            elif displaced_item == ObjType.BLUE:
                # Respawn food object
                self.maze[*pos] = ObjType.BLUE


        return displaced_item
    
    ## Helper functions


    ## Gym functions
    def render(self, fig_scale=1.0):
        """Show the entire as an image"""
        fig, ax = plt.subplots(figsize=(self.width * fig_scale, self.height * fig_scale))
        ax.imshow(self.maze, cmap=self.cmap, extent=(0, self.width, 0, self.height), vmin=0, vmax=4, interpolation="nearest")
        # Display Arrow for agent facing direction
        direction = rotate(self._agent_direction)
        pos = self._agent_pos + 0.5
        ax.annotate('', xy=(pos[1]+direction[1] * 0.5, self.height - (pos[0]+direction[0] * 0.5)), xytext=(pos[1], self.height - pos[0]), 
                        arrowprops=dict(arrowstyle="-|>", edgecolor="black", mutation_scale=100, facecolor="white"), xycoords="data")
        # TODO: If size is too big, don't display grid lines
        ax.set_xticks(np.arange(0, self.width, 1), labels=[])
        ax.set_yticks(np.arange(0, self.height, 1), labels=[])
        ax.grid(visible=True, color='gray', linewidth=1)
        plt.show()
        # return fig
    
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
        self._step_count = 0
        self._blue_count = 0
        self._green_count = 0
        self._reset_position()
        obs = self._get_observation()
        info = {
            'step_count': self._step_count,
            'agent_position': self.get_agent_position(),
            'green_count': self._green_count,
            'blue_count': self._blue_count,
            'item_displaced': None,
        }
        return obs, info

    def step(self, action: int) -> Tuple[int, float | None, bool, bool, dict]:
        self._step_count += 1
        # Calculate reward before taking action and getting obs
        obs = self._get_observation()
        # reward function based on obs and action
        reward = self._get_reward(obs, action)
        # Move the agent
        item_displaced = self._take_action(action)
        if item_displaced == ObjType.BLUE:
            self._blue_count += 1
        if item_displaced == ObjType.GREEN:
            self._green_count += 1

        obs = self._get_observation()
        # This environment will never be terminated, only truncated
        terminated = False
        truncated = self._step_count >= self.max_steps
        info = {
            'step_count': self._step_count, 
            'agent_position': self.get_agent_position(),
            'green_count': self._green_count,
            'blue_count': self._blue_count,
            'item_displaced': item_displaced,
        }
        return obs, reward, terminated, truncated, info

    ## Other functions

    def get_maze(self):
        return self.maze.copy()
    
    def get_agent_position(self):
        return np.array(self._agent_pos)

    @property
    def min_reward(self):
        return min(self.reward_function.values())

    @property
    def max_reward(self):
        return max(self.reward_function.values())

    @property
    def reward_list(self):
        return sorted(set(self.reward_function.values()))


