from .env import BaseMaze
from .maze import TMaze
from . import maze
from .spike_coding import SpikeCoder
from .collector import RewardCollector


ENV_DICT = {
    "t-maze": TMaze,
    "inverted-t": maze.InvertedTMaze,
    "random-maze": maze.RandomMaze,
    "custom-maze": maze.CustomMaze
}