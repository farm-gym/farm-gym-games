import os

from farmgym_games.game_agents.basic_agents import Farmgym_RandomAgent
from farmgym_games.game_builder.make_farm import make_farm
from farmgym_games.game_builder.run_farm import run_gym_xp


def env():
    yaml_path = os.path.join(os.path.dirname(__file__),"farm2.yaml")
    farm = make_farm(yaml_path)
    return farm

if __name__ == "__main__":
    agent = Farmgym_RandomAgent()
    run_gym_xp(env(), agent, max_steps=15, render="text")


