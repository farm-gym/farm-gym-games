import copy

from farmgym.v2.policy_api import Policy_API, Policy_helper
from farmgym_games.game_builder.run_farm import run_policy_xp
from farmgym_games.game_catalogue.farm1.farm import env as Farm1

farm = Farm1()
helper = Policy_helper(farm)

# Define single policies
water_soil_day1_5l = helper.create_water_soil(amount=5, delay=0, day=1)
herbicide_day2 = helper.create_scatter_cide(amount=5, delay=0, day=2)
fertilize_day4 = helper.create_scatter_fert(amount=5, delay=0, day=4)
water_soil_day6_1l = helper.create_water_soil(amount=1, delay=0, day=6)
harvest_fruit_delay4 = helper.create_harvest_fruit(delay=4)
harvest_ripe_delay4 = helper.create_harvest_ripe(delay=4)
# Note : some policies from old expert are currently missing :
#   - pesticide_day3
#   - sow_day5

# Combine policies
policies = [
    water_soil_day1_5l,
    herbicide_day2,
    fertilize_day4,
    water_soil_day6_1l,
    harvest_fruit_delay4,
    harvest_ripe_delay4,
]
combined_policy = Policy_API.combine_policies([policy.api for policy in policies])

# Run policy experiment
cumreward, cumcost = run_policy_xp(farm, copy.deepcopy(combined_policy), max_steps=200)
print(f"Combined policy run : cumreward = {cumreward}, cumcost = {cumcost}")
