from gymnasium.spaces import Tuple
from gymnasium.spaces import Box
from functools import partial

def flatten_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def wrapper(env):
    def farmgym_step(step, action):
        return step([action])
    obs, _ = env.reset()
    env.observation_space = Box(-1, 1, shape=(len(obs),))
    env.action_space = env.action_space.space 
    env.step = partial(farmgym_step, env.step)
    return env


def extract_values(tup):
    key, val = tup
    plant_stage = {
        "none": 0,
        "seed": 1,
        "entered_grow": 2,
        "grow": 3,
        "entered_bloom": 4,
        "bloom": 5,
        "entered_fruit": 6,
        "fruit": 7,
        "entered_ripe": 8,
        "ripe": 9,
        "entered_seed": 10,
        "harvested": 11,
        "dead": 12,
    }
    rain_amount = {
        "None": 0,
        "Light": 1,
        "Heavy": 2,
    }
    if key == "stage":
        return plant_stage[val]
    if key == "rain_amount":
        return rain_amount[val]
    if isinstance(val, dict):
        val = [v for k, v in val.items()]
    if val == 'False':
        return 0
    if val == 'True':
        return 1
    return val


def farmgym_to_gym_observations_flattened(farmgym_observations):
    gym_observations = []
    for fo in farmgym_observations:
        fa_key, fi_key, e_key, variable_key, path, value = fo
        gym_observations.append((variable_key, value))
    gym_observations = [extract_values(tup) for tup in gym_observations]
    gym_observations = flatten_list(gym_observations)
    return gym_observations
