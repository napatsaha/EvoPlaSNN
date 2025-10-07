from .env import BaseMaze
from .maze import TMaze
from . import maze
from .spike_coding import StateCoder
from .collector import RewardCollector


ENV_DICT: dict[str, env.BaseMaze] = {
    "t-maze": TMaze,
    "inverted-t": maze.InvertedTMaze,
    "random-maze": maze.RandomMaze,
    "custom-maze": maze.CustomMaze
}