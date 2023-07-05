from farmgym.v2.rendering.monitoring import make_variables_to_be_monitored

from farmgym_games.game_agents.basic_agents import Farmgym_RandomAgent
from farmgym_games.game_builder.run_farm import run_gym_xp
from farmgym_games.game_catalogue.farm1.farm import env as Farm1

farm = Farm1()
farm.add_monitoring(
    make_variables_to_be_monitored(
        [
            "f0.soil.available_Water#L",
            "f0.soil.available_N#g",
            "f0.soil.microlife_health_index#%",
            "f0.plant.pollinator_visits#nb",
            "f0.plant.size#cm",
            "f0.plant.flowers_per_plant#nb",
            "f0.plant.flowers_pollinated_per_plant#nb",
            "f0.plant.cumulated_water#L",
            "f0.plant.cumulated_stress_water#L",
            "f0.plant.cumulated_nutrients_N#g",
            "f0.plant.cumulated_stress_nutrients_N#g",
            "f0.plant.fruits_per_plant#nb",
            "f0.plant.fruit_weight#g",
            "f0.weeds.seeds#nb",
            "f0.weeds.grow#nb",
            "f0.weeds.flowers#nb.mat",
            "f0.weeds.flowers#nb",
            "f0.cide.amount#kg",
            "f0.pests.plot_population#nb",
            "f0.pests.onplant_population#nb[plant]",
            "f0.pests.onplant_population#nb[weeds]",
        ]
    )
)
agent = Farmgym_RandomAgent()
run_gym_xp(farm, agent, max_steps=100)

## Tensorboard monitoring can be sped up by disabling Matrix view, since images in tensorboard can reduce the performance.
## This can be done by adding matview=False in add_monitoring function

# farm = Farm1()
# farm.add_monitoring(
# 		make_variables_to_be_monitored(
# 			[
# 				"f0.soil.available_Water#L",
# 				"f0.soil.available_N#g",
# 				"f0.soil.microlife_health_index#%",
# 				"f0.plant.pollinator_visits#nb",
# 				"f0.plant.size#cm",
# 				"f0.plant.flowers_per_plant#nb",
# 				"f0.plant.flowers_pollinated_per_plant#nb",
# 				"f0.plant.cumulated_water#L",
# 				"f0.plant.cumulated_stress_water#L",
# 				"f0.plant.cumulated_nutrients_N#g",
# 				"f0.plant.cumulated_stress_nutrients_N#g",
# 				"f0.plant.fruits_per_plant#nb",
# 				"f0.plant.fruit_weight#g",
# 				"f0.weeds.seeds#nb",
# 				"f0.weeds.grow#nb",
# 				"f0.weeds.flowers#nb.mat",
# 				"f0.weeds.flowers#nb",
# 				"f0.cide.amount#kg",
# 				"f0.pests.plot_population#nb",
# 				"f0.pests.onplant_population#nb[plant]",
# 				"f0.pests.onplant_population#nb[weeds]",
# 			]
# 		)
# 	,matview=False)
# agent = Farmgym_RandomAgent()
# run_gym_xp(farm, agent, max_steps=100)


## Monitoring uses Tensorboard by default, but can be replaced by Matplotlib,
## by disabling Tensorboard

# farm = Farm1()
# farm.add_monitoring(
# 		make_variables_to_be_monitored(
# 			[
# 				"f0.soil.available_Water#L",
# 				"f0.soil.available_N#g",
# 				"f0.soil.microlife_health_index#%",
# 				"f0.plant.pollinator_visits#nb",
# 				"f0.plant.size#cm",
# 				"f0.plant.flowers_per_plant#nb",
# 				"f0.plant.flowers_pollinated_per_plant#nb",
# 				"f0.plant.cumulated_water#L",
# 				"f0.plant.cumulated_stress_water#L",
# 			]
# 		)
# 	, tensorboard=False)

# agent = Farmgym_RandomAgent()
# run_gym_xp(farm, agent, max_steps=100)
