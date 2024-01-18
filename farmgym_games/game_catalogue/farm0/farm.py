import os

from farmgym_games.game_builder.make_farm import make_farm
from farmgym_games.game_builder.run_farm import run_policy_xp
from farmgym.v2.policy_api import Policy_API, Policy_helper

def env():
    yaml_path = os.path.join(os.path.dirname(__file__), "farm0.yaml")
    farm = make_farm(yaml_path)    
    helper = Policy_helper(farm)
    farm.helper = helper
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
    policies = farm.helper.get_random_policies(5)
    scores = {name:0 for name, policy in policies}
    # For each policy
    for name, policy in policies:
        # Do 10 runs
        for i in range(10):
            r, c = run_policy_xp(farm, policy)
            # Add current reward to policy score 
            scores[name] += r
    print([(key, f"{score/10:.3f}") for key,score in scores.items()])