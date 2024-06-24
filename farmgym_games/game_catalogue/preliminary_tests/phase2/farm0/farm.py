import os

from farmgym_games.game_builder.make_farm import make_farm
from farmgym_games.game_builder.run_farm import run_policy_xp
from farmgym.v2.policy_api import Policy_API, Policy_helper
from farmgym.v2.rendering.monitoring import make_variables_to_be_monitored

def env():
    yaml_path = os.path.join(os.path.dirname(__file__), "farm.yaml")
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

    #To be removed once the parameter tuning of entities is done:
    farm.add_monitoring(
        make_variables_to_be_monitored(
            [
                "f0>weather>rain_amount#mm.day-1",
                "f0>weather>clouds#%",
                "f0>weather>air_temperature>mean#°C",
                "f0>weather>wind>speed#km.h-1",
                "f0>soil>available_Water#L",
                "f0>soil>microlife_health_index#%",
                "f0>soil>available_N#g",
                "f0>soil>available_P#g",
                "f0>soil>available_K#g",
                "f0>soil>available_C#g",
                "f0>plant>size#cm",
                "f0>plant>cumulated_stress_water#L",
                "f0>plant>cumulated_stress_nutrients_N#g",
                "f0>plant>cumulated_stress_nutrients_P#g",
                "f0>plant>cumulated_stress_nutrients_K#g",
                "f0>plant>cumulated_stress_nutrients_C#g",
                "f0>plant>cumulated_nutrients_N#g",
                "f0>plant>cumulated_nutrients_P#g",
                "f0>plant>cumulated_nutrients_K#g",
                "f0>plant>cumulated_nutrients_C#g",
                # "f0>plant>flowers_per_plant#nb@mat",
                "f0>plant>flowers_per_plant#nb",
                "f0>plant>fruits_per_plant#nb",
                "f0>plant>fruit_weight#g",
                "f0>plant>stage@name",
            ]
        )
    )

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