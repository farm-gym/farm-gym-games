from farmgym.v2.games.rungame import Farmgym_RandomAgent, run_gym_xp
from farmgym.v2.games.make_farm import make_farm
import os

def env():
    yaml_path = os.path.join(os.path.dirname(__file__),"farm0.yaml")
    farm = make_farm(yaml_path)
    return farm

if __name__ == "__main__":
    agent = Farmgym_RandomAgent()
    run_gym_xp(env(), agent, max_steps=15, render="text")


