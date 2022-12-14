import gym
from gym import spaces
from gym.utils.step_api_compatibility import step_api_compatibility 
import farmgym_games.farm1.farm as cb
import numpy as np
import time
import os
from farmgym_games.utils import (
    farmgymobs_to_obs,
    update_farm_writer,
    observation_hide_final_state_of_plants,
)


class Farm1(gym.Env):
    """
    Farm1 is a difficult 1x1 farm with only one possible plant : beans, planted in a clay ground.
    The advised maximum episode length is 365 (as in 365 days).

    The Farm has the weather of Lille in France (e.g. well suited for the culture of beans), the initial day is 1. Initially the field is healthy and contains all the nutrient necessary to the plant.

    The reward is the number of grams of harvested beans, and there is a negative reward for very low microlife in soil (due to pesticides).

    The condition for end of episode (self.step returns done) is that the day is >= 365 or that the plant is dead.

    Parameters
    ----------

    api_compatibility: False
        If true apply step api compatibility to gym version 0.21. See https://www.gymlibrary.dev/api/utils/#gym.utils.step_api_compatibility.step_api_compatibility.

    Notes
    -----
    State:
        The state consists of
    
        * Day (from 1 to 365)
        * mean air temperature (°C)
        * min air temperature (°C)
        * max air temperature (°C)
        * rain amount (mm)
        * sun-exposure (from 1 to 5)
        * consecutive dry day (int)
        * stage of growth of the plant (int)
        * size of the plant in cm.
        * Soil wet_surface#m2.day-1
        * fertilizer amount#kg
        * Pests plot_population#nb
        * Pollinators occurrence#bin
        * Weeds grow#nb
        * Weeds flowers#nb
        * Weight of fruits (g)
        * Microlife health index (%)

    Actions:
        The actions are :
    
        * doing nothing.
        * 2 levels of watering the field (1L or 5L of water)
        * harvesting
        * sow some seeds
        * scatter fertilizer
        * scatter herbicide
        * scatter pesticide
        * remove weeds by hand
    """

    name = "Farm1"

    observations_txt = [
        "Day (from 1 to 365)",
        "Mean air temperature (°C)",
        "Min air temperature (°C)",
        "Max air temperature (°C)",
        "Rain amount",
        "Sun-exposure (from 1 to 5)",
        "Consecutive dry day (int)",
        "Stage of growth of the plant",
        "Number of fruits (int)",
        "Size of the plant in cm",
        "Soil wet_surface (m2.day-1)",
        "fertilizer amount (kg)",
        "Pollinators occurrence (bin)",
        "Weeds grow (nb)",
        "Weeds flowers (nb)",
        "weight of fruits",
        "microlife health index (%)",
    ]

    def __init__(self, api_compatibility = False):
        # init base classes
        gym.Env.__init__(self)
        self.api_compatibility = api_compatibility

        self.farm = cb.env()
        self.farm.gym_step([])

        # observation and action spaces
        # Day, temp mean, temp min, temp max, rain amount, sun exposure, consecutive dry day, stage, size#cm, nb of fruits,
        # wet surface,  fertilizer amount,  pollinators occurrence, weeds grow nb, weeds flower nb, weight of fruits, microlife health index %
        high = np.array(
            [365, 50, 50, 50, 300, 10, 10, 10, 100, 300, 10, 10, 1, 100, 100, 5000, 100]
        )
        low = np.array([0, -50, -50, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Discrete(9)

        # monitoring writer
        params = {}
        self.iteration = 0

        # initialize
        self.state = None
        self.reset()

    def reset(self):
        observation = self.farm.gym_reset()[0]
        self.farm.gym_step([])
        if self.api_compatibility:
            return observation_hide_final_state_of_plants(
                farmgymobs_to_obs(observation), id_of_plants_stage=7
            )
        else:
                        return observation_hide_final_state_of_plants(
                farmgymobs_to_obs(observation), id_of_plants_stage=7
            ), {}


    def step(self, action):
        # Stepping
        #   farmgym run with a cycle of 2 steps: 1 (empty) step of getting observation ("morning"), then 1 step of acting ("afternoon").
        #   Classic RL methodology use only 1 step : performing an action return the next observation
        #   To match this 2, we run the 'farmgy observation step ("morning")' right after the action.
        #   With this method, it will be like classic RL 'step' for the user
        _, reward, is_done, info = self.farm.farmgym_step(self.num_to_action(action))
        obs1, _, _, info = self.farm.gym_step([])

        if hasattr(reward, "__len__"):
            reward = reward[0]

        if np.array(obs1[-1]).item() < 10:
            reward -= 2  # if microlife is < 10%, negative reward

        observation = observation_hide_final_state_of_plants(
            farmgymobs_to_obs(obs1), id_of_plants_stage=7
        )
        if self.api_compatibility:
            return step_api_compatibility((observation, reward, False, is_done, info), output_truncation_bool=False)
        else:
            return observation, reward, False, is_done, info

    def num_to_action(self, num):
        if num == 1:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Soil-0",
                    "water_discrete",
                    {"plot": (0, 0), "amount#L": 1, "duration#min": 60},
                )
            ]
        elif num == 2:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Soil-0",
                    "water_discrete",
                    {"plot": (0, 0), "amount#L": 5, "duration#min": 60},
                )
            ]
        elif num == 3:
            return [("BasicFarmer-0", "Field-0", "Plant-0", "harvest", {})]
        elif num == 4:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Plant-0",
                    "sow",
                    {"plot": (0, 0), "amount#seed": 1, "spacing#cm": 10},
                )
            ]
        elif num == 5:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Fertilizer-0",
                    "scatter_bag",
                    {"plot": (0, 0), "amount#bag": 1},
                )
            ]
        elif num == 6:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Cide-0",
                    "scatter_bag",
                    {"plot": (0, 0), "amount#bag": 1},
                )
            ]
        elif num == 7:
            return [
                (
                    "BasicFarmer-0",
                    "Field-0",
                    "Cide-1",
                    "scatter_bag",
                    {"plot": (0, 0), "amount#bag": 1},
                )
            ]
        elif num == 8:
            return [("BasicFarmer-0", "Field-0", "Weeds-0", "remove", {"plot": (0, 0)})]
        else:
            return []  # Do nothing.
