from farmgym_games.game_catalogue.farm1.farm import env as Farm1
from farmgym.v2.rendering.monitoring import make_variables_to_be_monitored

from farmgym_games.game_agents.basic_agents import Farmgym_RandomAgent
from farmgym_games.game_builder.run_farm import run_policy_xp

from farmgym.v2.policy_api import Policy_helper, Policy_API
import copy

farm = Farm1()
helper = Policy_helper(farm)

## Get policies will generate the default policies for all present entities in the farm
policies = helper.get_policies()
print(f"\nAll policies : {policies}\n")

## It is possible also to select only policies for a specific entity 
plant_policies = helper.get_plant_policies()
weeds_policies = helper.get_weeds_policies()
soil_policies = helper.get_soil_policies()
fertilizer_policies = helper.get_fertilizer_policies()
facility_policies = helper.get_facility_policies()
print(f"plant_policies = {plant_policies}")
print(f"weeds_policies = {weeds_policies}")
print(f"soil_policies = {soil_policies}")
print(f"fertilizer_policies = {fertilizer_policies}")
print(f"facility_policies = {facility_policies}\n")

## Each policy in this case is a wrapper around a policy api (.api) and policy parameters (.infos)
print(f"{facility_policies[0]}.api : {facility_policies[0].api}")
print(f"{facility_policies[0]}.infos() : {facility_policies[0].infos()}")

## To create a custom policy with different parameters, we can do the following:
water_soil = helper.create_water_soil(amount=5, delay=3)

# It is possible to combine policy_apis with the combine method
combined = Policy_API.combine_policies([policies[0].api, policies[1].api, policies[2].api, policies[3].api])
# or using addition operator
combined = policies[0].api + policies[1].api + policies[2].api + policies[3].api

# After combining the policies, we can run the obtained policy on Farm1 for example: 
cumreward, cumcost = run_policy_xp(farm, copy.deepcopy(combined), max_steps=150)
print(f"Combined policy run : cumreward = {cumreward}, cumcost = {cumcost}")