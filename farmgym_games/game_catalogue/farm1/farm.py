import os
import random 
import numpy as np

from farmgym_games.game_agents.basic_agents import Farmgym_RandomAgent
from farmgym_games.game_builder.make_farm import make_farm
from farmgym_games.game_builder.run_farm import run_gym_xp, run_policy_xp

from farmgym.v2.policy_api import Policy_API, Policy_helper
def env():
    yaml_path = os.path.join(os.path.dirname(__file__), "farm1.yaml")
    farm = make_farm(yaml_path)
    return farm


if __name__ == "__main__":
    # agent = Farmgym_RandomAgent()
    # run_gym_xp(env(), agent, max_steps=15, render="text")
    
    farm = env()
    helper = Policy_helper(farm)

    # Get params
    # print(helper.get_soil_params())
    # print(helper.get_weeds_params())
    # print(helper.get_fertilizer_params())
    
    # Get policies
    #print(helper.get_soil_policies())
    #print(helper.get_weeds_policies())
    #print(helper.get_fertilizer_policies())
    
    # params0 = helper.get_soil_params()
    # #fn0 = helper.get_soil_policies()
    # params1 = helper.get_weeds_params()
    # #fn1 = helper.get_weeds_policies()
    # params2 = helper.get_fertilizer_params()
    # #fn2 = helper.get_fertilizer_policies()
    
    policies = helper.get_random_policies(10)
    scores = {}
    for n,p in policies:
        # Do 100 runs
        score = []
        for i in range(10):
            r, c = run_policy_xp(farm, p)
            # Add current reward to policy score 
            score.append(r)
        #print(score)
        scores[n] = np.mean(score)
    print("___")
    for k in scores.keys():
        print(f"{k} : {scores[k]}")