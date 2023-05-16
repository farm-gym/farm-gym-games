from farmgym.v2.games.rungame import Farmgym_RandomAgent, run_gym_xp
from farmgym.v2.games.make_farm import make_farm

if __name__ == "__main__":
    farm = make_farm("farm1.yaml")
    agent = Farmgym_RandomAgent()
    run_gym_xp(farm, agent, max_steps=15, render="text")


