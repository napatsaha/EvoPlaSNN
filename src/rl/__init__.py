from .env import BaseMaze
from .maze import TMaze
from . import maze
from .spike_coding import StateCoder
from .collector import RewardCollector


ENV_DICT: dict[str, BaseMaze] = {
    "t-maze": TMaze,
    "adv-t-maze": maze.AdvTMaze,
    "inverted-t": maze.InvertedTMaze,
    "random-maze": maze.RandomMaze,
    "custom-maze": maze.CustomMaze
}