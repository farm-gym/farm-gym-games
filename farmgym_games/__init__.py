from .game0_env import Farm0
from .game1_env import Farm1
import gym

gym.envs.register(
     id='Farm0-v0',
     entry_point='farmgym_games.game0_env:Farm0',
     max_episode_steps=365,
)

gym.envs.register(
     id='Farm1-v0',
     entry_point='farmgym_games.game1_env:Farm1',
     max_episode_steps=365,
)
