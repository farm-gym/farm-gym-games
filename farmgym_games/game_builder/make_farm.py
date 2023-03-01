
import farmgym

from farmgym.v2.farm import Farm
from farmgym.v2.field import Field
from farmgym.v2.farmers.BasicFarmer import BasicFarmer
from farmgym.v2.scorings.BasicScore import BasicScore
from farmgym.v2.rules.BasicRule import BasicRule
#from farmgym.v2.policy_api import Policy_API
import inspect





def make_basicfarm(name, field, entities, init_values=None, farmers=[{"max_daily_interventions":1}]):
    #farm_call = " ".join(inspect.stack()[1].code_context[0].split("=")[0].split())
    filep = "/".join(inspect.stack()[1].filename.split("/")[0:-1])


    name_score = name + "_score.yaml"
    name_init = name + "_init.yaml"
    name_actions = name + "_actions.yaml"
    entities1 = []
    for e, i in entities:
        entities1.append((e, i))

    field1 = Field(
        localization=field["localization"],
        shape=field["shape"],
        entity_managers=entities1,
    )

    #farmer1 = BasicFarmer(max_daily_interventions=1)
    ffarmers = [BasicFarmer(max_daily_interventions=f["max_daily_interventions"]) for f in farmers]
    scoring = BasicScore(score_configuration=filep + "/"+ name_score)
    #scoring = BasicScore(score_configuration=CURRENT_DIR / name_score)

    free_observations = []
    free_observations.append(("Field-0", "Weather-0", "day#int365", []))
    free_observations.append(("Field-0", "Weather-0", "air_temperature", []))

    terminal_CNF_conditions = [
        [(("Field-0", "Weather-0", "day#int365", []), lambda x: x.value, ">=", 360)],
        [
            (
                ("Field-0", "Plant-0", "global_stage", []),
                lambda x: x.value,
                "in",
                ["dead", "harvested"],
            )
        ],
    ]
    rules = BasicRule(
        init_configuration=filep + "/"+ name_init,
        actions_configuration=filep + "/"+ name_actions,
        terminal_CNF_conditions=terminal_CNF_conditions,
        free_observations=free_observations,
        initial_conditions_values=init_values    )

    # DEFINE one policy:
    policies = []

    farm = Farm(
        fields=[field1],
        farmers=ffarmers,
        scoring=scoring,
        rules=rules,
        policies=policies,
    )
    farm.name = name
    return farm
