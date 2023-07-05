import numpy as np


def farmgymobs_to_obs(obs_lst):
    """
    Extract all the numerical values of a farmgym obs and return a flat array.
    """
    res = np.array([])
    for obs in obs_lst:
        if type(obs) is dict:
            to_add = np.array(list(obs.values()))
        else:
            to_add = np.array([obs])

        res = np.concatenate((res, to_add), axis=None)
    return res


def update_farm_writer(writer, monitor_variables, farm, iteration):
    """
    Update the farm writer with the valued monitored by the farm.
    """
    for i in range(len(monitor_variables)):
        v = monitor_variables[i]
        fi_key, entity_key, var_key, map_v, name_to_display, v_range = v
        value = map_v(farm.fields[fi_key].entities[entity_key].variables[var_key])
        writer.add_scalar(name_to_display, np.round(value, 3), iteration)


def observation_hide_final_state_of_plants(obs, id_of_plants_stage):
    """
    Update the plants 'stage of growth' in observations to hide
    when the fruit is ready to be harvested
    """
    if obs[id_of_plants_stage] in [6, 7, 8, 9]:
        obs[id_of_plants_stage] = 6
    return obs


def get_last_monitor_values(writer):
    global_step = writer.data["global_step"].max()
    return writer.data.loc[writer.data["global_step"] == global_step, ["tag", "value"]]


def get_desc_from_value(id_to_desc, item_name_to_desc):
    # self.env.farm.fields.entities.Plant-0.variables.global_stage
    plant_stage = {
        0: "none",
        1: "seed",
        2: "entered_grow",
        3: "grow",
        4: "entered_bloom",
        5: "bloom",
        6: "entered_fruit",
        7: "fruit",
        8: "entered_ripe",
        9: "ripe",
        10: "entered_seed",
        11: "harvested",
        12: "dead",
    }

    # self.env.farm.fields.entities.Weather-0.variables.rain_amount
    rain_amount = {
        0: "None",
        1: "Light",
        2: "Heavy",
    }

    desc = ""

    if item_name_to_desc == "rain_amount":
        desc = rain_amount[id_to_desc]
    elif item_name_to_desc == "plant_stage":
        desc = plant_stage[id_to_desc]
    return desc
