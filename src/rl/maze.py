from typing import override
from .env import BaseMaze
import numpy as np


class TMaze(BaseMaze):
    """
    T-Maze environment for reinforcement learning.
    """
    @override
    def _create_maze(self):
        # Create empty array (filled with walls, as 1's)
        self.maze = np.full((self.height, self.width), dtype=np.int8, fill_value=self.WALL)
        # Add traversing paths (as 0's)
        mid_width = self.width // 2
        self.maze[:, mid_width] = self.EMPTY
        self.maze[0, :] = self.EMPTY
        # Set agent starting position
        self.maze[-1, mid_width] = self.AGENT
        # Set good position at right wing of T
        self.maze[0, -1] = self.GOOD
        # Set bad position at left wing of T
        self.maze[0, 0] = self.BAD


class InvertedTMaze(BaseMaze):
    """
    T-Maze environment but with the T on the bottom.
    """
    @override
    def _create_maze(self):
        # Create empty array (filled with walls, as 1's)
        self.maze = np.full((self.height, self.width), dtype=np.int8, fill_value=self.WALL)
        # Add traversing paths (as 0's)
        mid_width = self.width // 2
        self.maze[:, mid_width] = self.EMPTY
        self.maze[-1, :] = self.EMPTY
        # Set agent starting position
        self.maze[0, mid_width] = self.AGENT
        # Set good position at right wing of T
        self.maze[-1, 0] = self.GOOD
        # Set bad position at left wing of T
        self.maze[-1, -1] = self.BAD


class RandomMaze(BaseMaze):
    COLOR_LIST = ["black", "white", "blue", "green", "red"]
    def __init__(self, size=None, width=None, height=None, pad=1, *, 
                 num_obs: int = 5,
                 **kwargs):
        self.num_obs = num_obs
        super().__init__(size, width, height, pad, **kwargs)

    @override
    def _create_maze(self):
        # Create empty array (filled with empty spaces)
        self.maze = np.full((self.height, self.width), dtype=np.int8, fill_value=self.EMPTY)
        
        # Randomly allocate non-empty cells
        obj_flat_idx = np.random.choice(np.arange(self.area), size=self.num_obs + 3, replace=False)
        # Assign the first num_obs cells to obstacles / wall
        wall_flat_idx = obj_flat_idx[:self.num_obs]
        # The remaining 3 cells will be the good item, bad item and agent
        good_flat_idx = obj_flat_idx[self.num_obs]
        bad_flat_idx = obj_flat_idx[self.num_obs + 1]
        agent_flat_idx = obj_flat_idx[self.num_obs + 2]
        # Set obstacles
        self.maze[np.unravel_index(wall_flat_idx, self.maze.shape)] = self.WALL
        # Set agent starting position
        self.maze[np.unravel_index(agent_flat_idx, self.maze.shape)] = self.AGENT
        # Set good position 
        self.maze[np.unravel_index(good_flat_idx, self.maze.shape)] = self.GOOD
        # Set bad position 
        self.maze[np.unravel_index(bad_flat_idx, self.maze.shape)] = self.BAD


class CustomMaze(BaseMaze):
    """
    A Custome Maze where an array of integers can be passed to create such a maze.  
    Positions for Agent, Good and Bad items must be specified with 2, 3 and 4 respectively.
    """
    def __init__(self, maze_file: str, pad=1, **kwargs):
        self.maze_file = maze_file
        super().__init__(pad=pad, **kwargs)

    @override
    def _create_maze(self):
        # Load maze from file
        loaded_maze = np.loadtxt(self.maze_file, dtype=np.int8)
        if loaded_maze.ndim != 2:
            raise ValueError("Loaded maze must be a 2D array.")
        if not ((loaded_maze == self.WALL) | (loaded_maze == self.EMPTY) | 
                (loaded_maze == self.AGENT) | (loaded_maze == self.GOOD) | 
                (loaded_maze == self.BAD)).all():
            raise ValueError(f"Maze can only contain values {self.WALL} (wall), {self.EMPTY} (empty), {self.AGENT} (agent), {self.GOOD} (good), {self.BAD} (bad).")
        if np.sum(loaded_maze == self.AGENT) != 1:
            raise ValueError("Maze must contain exactly one agent starting position.")
        if np.sum(loaded_maze == self.GOOD) != 1:
            raise ValueError("Maze must contain exactly one good item.")
        if np.sum(loaded_maze == self.BAD) != 1:
            raise ValueError("Maze must contain exactly one bad item.")
        
        self.maze = loaded_maze
        self.height, self.width = self.maze.shape
        self.area = self.height * self.width
