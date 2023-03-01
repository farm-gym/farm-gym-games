from farmgym.v2.farm import Farm
from farmgym.v2.field import Field
from farmgym.v2.farmers.BasicFarmer import BasicFarmer
from farmgym.v2.scorings.BasicScore import BasicScore
from farmgym.v2.rules.BasicRule import BasicRule
from farmgym.v2.entities.Weather import Weather
from farmgym.v2.entities.Soil import Soil
from farmgym.v2.entities.Plant import Plant

from farmgym.v2.entities.Birds import Birds
from farmgym.v2.entities.Facilities import Facility
from farmgym.v2.entities.Fertilizer import Fertilizer
from farmgym.v2.entities.Weeds import Weeds
from farmgym.v2.entities.Pests import Pests
from farmgym.v2.entities.Pollinators import Pollinators
from farmgym.v2.entities.Cide import Cide


from farmgym.v2.rendering.monitoring import mat2d_value, sum_value


from gym.envs.registration import register

import os
from pathlib import Path

file_path = Path(os.path.realpath(__file__))
CURRENT_DIR = file_path.parent


def env():
    ##########################################################################
    entities1 = []
    entities1.append((Weather, "lille"))
    entities1.append((Soil, "loam"))
    entities1.append((Plant, "bean"))

    entities1.append((Fertilizer, "slow_all"))

    entities1.append((Pollinators, "bee"))

    entities1.append((Weeds, "base_weed"))
    entities1.append((Cide, "herbicide"))

    entities1.append((Pests, "basic"))
    entities1.append((Pollinators, "bee"))
    entities1.append((Cide, "pesticide"))

    field1 = Field(
        localization={"latitude#°": 50.38, "longitude#°": 3.03, "altitude#m": 10},
        shape={"length#nb": 1, "width#nb": 1, "scale#m": 1.0},
        entity_managers=entities1,
    )

    farmer1 = BasicFarmer(max_daily_interventions=1)
    scoring = BasicScore(score_configuration=CURRENT_DIR / "farm_score.yaml")

    free_observations = []
    free_observations.append(("Field-0", "Weather-0", "day#int365", []))
    free_observations.append(("Field-0", "Weather-0", "air_temperature", []))
    free_observations.append(("Field-0", "Weather-0", "rain_amount", []))
    free_observations.append(("Field-0", "Weather-0", "sun_exposure#int5", []))
    free_observations.append(("Field-0", "Weather-0", "consecutive_dry#day", []))

    free_observations.append(("Field-0", "Plant-0", "stage", []))
    free_observations.append(("Field-0", "Plant-0", "fruits_per_plant#nb", []))

    free_observations.append(("Field-0", "Plant-0", "size#cm", []))

    free_observations.append(("Field-0", "Soil-0", "wet_surface#m2.day-1", []))

    free_observations.append(("Field-0", "Fertilizer-0", "amount#kg", []))
    free_observations.append(("Field-0", "Pollinators-0", "occurrence#bin", []))
    free_observations.append(("Field-0", "Weeds-0", "grow#nb", []))
    free_observations.append(("Field-0", "Weeds-0", "flowers#nb", []))
    free_observations.append(("Field-0", "Plant-0", "fruit_weight#g", []))

    free_observations.append(("Field-0", "Soil-0", "microlife_health_index#%", []))

    terminal_CNF_conditions = [
        [(("Field-0", "Weather-0", "day#int365", []), lambda x: x.value, ">=", 360)],
        [(("Field-0", "Plant-0", "global_stage", []), lambda x: x.value, "==", "harvested")],
        [(("Field-0", "Plant-0", "global_stage", []), lambda x: x.value, "==", "dead")],
    ]
    rules = BasicRule(
        init_configuration=CURRENT_DIR / "farm_init.yaml",
        actions_configuration=CURRENT_DIR / "farm_actions.yaml",
        terminal_CNF_conditions=terminal_CNF_conditions,
        free_observations=free_observations,
    )

    farm = Farm(fields=[field1], farmers=[farmer1], scoring=scoring, rules=rules)

    ##########################################################################

    # var = []

    # var.append(
    #     (
    #         "Field-0",
    #         "Soil-0",
    #         "available_N#g",
    #         lambda x: sum_value(x),
    #         "Available Nitrogen (g)",
    #         "range_auto",
    #     )
    # )
    # var.append(
    #     (
    #         "Field-0",
    #         "Soil-0",
    #         "available_Water#L",
    #         lambda x: sum_value(x),
    #         "Available Water (g)",
    #         "range_auto",
    #     )
    # )
    # var.append(
    #     (
    #         "Field-0",
    #         "Plant-0",
    #         "size#cm",
    #         lambda x: sum_value(x),
    #         "Size (cm)",
    #         "range_auto",
    #     )
    # )
    # var.append(
    #     (
    #         "Field-0",
    #         "Plant-0",
    #         "flowers_per_plant#nb",
    #         lambda x: sum_value(x),
    #         "Flowers (nb)",
    #         "range_auto",
    #     )
    # )
    # var.append(
    #     (
    #         "Field-0",
    #         "Plant-0",
    #         "flowers_pollinated_per_plant#nb",
    #         lambda x: sum_value(x),
    #         "Flowers pollinated (nb)",
    #         "range_auto",
    #     )
    # )

    # var.append(
    #     (
    #         "Field-0",
    #         "Plant-0",
    #         "fruit_weight#g",
    #         lambda x: sum_value(x),
    #         "Fruits weight (g)",
    #         "range_auto",
    #     )
    # )

    # var.append(
    #     (
    #         "Field-0",
    #         "Pests-0",
    #         "plot_population#nb",
    #         lambda x: x[(0, 0)].value,
    #         "Insects (nb)",
    #         "range_auto",
    #     )
    # )
    # var.append(
    #     (
    #         "Field-0",
    #         "Soil-0",
    #         "amount_cide#g",
    #         lambda x: x["soil"][0, 0].value,
    #         "Herbicide + pesticide in soil",
    #         "range_auto",
    #     )
    # )
    # var.append(
    #     (
    #         "Field-0",
    #         "Weeds-0",
    #         "seeds#nb",
    #         lambda x: x[0, 0].value,
    #         "Weeds seed nb",
    #         "range_auto",
    #     )
    # )

    # farm.add_monitoring(var)
    # farm.monitor_variables = var

    return farm


if __name__ == "__main__":
    from farmgym.v2.games.rungame import run

    run(env(), max_steps=100, render=False, monitoring=True)
