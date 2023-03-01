from farmgym_games.game_catalogue.game0_env import Farm0
from farmgym_games.game_catalogue.game1_env import Farm1
from game_builder.utils import  farmgymobs_to_obs, get_desc_from_value
import  gym

gym.envs.register(
     id='Farm0-v0',
     entry_point='farmgym_games.game0_env:Farm0',
)

gym.envs.register(
     id='OldV21Farm0-v0',
     entry_point='farmgym_games.game0_env:Farm0',
    kwargs={"api_compatibility":True}
)

gym.envs.register(
     id='Farm1-v0',
     entry_point='farmgym_games.game1_env:Farm1',
)

gym.envs.register(
     id='OldV21Farm1-v0',
     entry_point='farmgym_games.game1_env:Farm1',
    kwargs={"api_compatibility":True}
)

gym.envs.register(
     id='Farm2-v0',
     entry_point='farmgym_games.game2_env:Farm2',
)

gym.envs.register(
     id='OldV21Farm2-v0',
     entry_point='farmgym_games.game2_env:Farm2',
    kwargs={"api_compatibility":True}
)

