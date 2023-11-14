import os

from farmgym_games.game_agents.basic_agents import Farmgym_RandomAgent
from farmgym_games.game_builder.make_farm import make_farm
from farmgym_games.game_builder.run_farm import run_gym_xp, run_policy_xp
from farmgym.v2.policy_api import Policy_API, Policy_helper

def env():
    yaml_path = os.path.join(os.path.dirname(__file__), "farm0.yaml")
    farm = make_farm(yaml_path)
    helper = Policy_helper(farm)
    
    policies = {
        "p1": Policy_API.combine_policies([policy.api for policy in helper.get_policies(frequency=1)]),
        "p3": Policy_API.combine_policies([policy.api for policy in helper.get_policies(frequency=3)]),
        "p5": Policy_API.combine_policies([policy.api for policy in helper.get_policies(frequency=5)]),
        "p7": Policy_API.combine_policies([policy.api for policy in helper.get_policies(frequency=7)]),
    }
    farm.policies = policies


    return farm

if __name__ == "__main__":
    farm = env()
    # Init scores for all policies
    policies = farm.policies
    scores = {p:0 for p in policies.keys()}
    # For each policy
    for idx in policies:
        # Do 100 runs
        for i in range(100):
            r, c = run_policy_xp(farm, policies[idx])
            # Add current reward to policy score 
            scores[idx] += r
    print(scores)

