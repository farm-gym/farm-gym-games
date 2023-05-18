import gym
from gym import spaces
from gym.utils.step_api_compatibility import step_api_compatibility 
import farmgym_games.game_catalogue.farm0.farm as cb
from farmgym_games.game_builder.utils import (
    farmgymobs_to_obs,
    observation_hide_final_state_of_plants,
)
import numpy as np


class Farm0(gym.Env):
    """
    Farm0 is a very basic 1x1 farm with only one possible plant : tomato, planted in a clay ground.
    The actions are to water the field or to harvest it.
    The advised maximum episode length is 365 (as in 365 days).

    The Farm has the weather of Montpellier in France (e.g. fairly warm weather, well suited for the culture of tomato), the initial day is 100. Initially the field is healthy and contains all the nutrient necessary to the plant.

    The reward is the number of grams of harvested tomatoes.

    The condition for end of episode (self.step returns done) is that the day is >= 365 or that the field has been harvested, or that the plant is dead.

    Parameters
    ----------
    api_compatibility: boolean, default = False
        Apply compatibility to v0.21 gym api.
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
        * size of the plant in cm
        * number of fruits (int)
        * weight of fruits (g).

    Actions:
        The actions are :
    
        * doing nothing.
        * 2 levels of watering the field (1L or 5L of water)
        * harvesting
    """

    name = "Farm0"

    observations_txt = [
        "Day (from 1 to 365)",
        "Mean air temperature (°C)",
        "Min air temperature (°C)",
        "Max air temperature (°C)",
        "Rain amount",
        "Sun-exposure (from 1 to 5)",
        "Consecutive dry day (int)",
        "Stage of growth of the plant",
        "Size of the plant in cm",
        "nb of fruits",
        "weight of fruits",
    ]

    def __init__(self, api_compatibility=False):
        # init base classes
        gym.Env.__init__(self)
        self.api_compatibility = api_compatibility

        self.farm = cb.env()
        self.farm.gym_step([])
        # observation and action spaces
        # Day, temp mean, temp min, temp max, rain amount, sun exposure, consecutive dry day, stage, size#cm, fruit weight, nb of fruits, weights
        high = np.array([365, 50, 50, 50, 300, 7, 100, 10, 200, 100, 5000])
        low = np.array([0, -50, -50, -50, 0, 0, 0, 0, 0, 0, 0])
        self.n_obs = len(high)
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Discrete(4)

        # monitoring writer
        params = {}
        self.iteration = 0

        # initialize
        self.state = None
        self.reset()

    def reset(self):
        observation = self.farm.gym_reset()[0]
        obs1, _, _, info = self.farm.gym_step([])
        if self.api_compatibility:
            return observation_hide_final_state_of_plants(
                farmgymobs_to_obs(obs1), id_of_plants_stage=7
            )
        else:
            return observation_hide_final_state_of_plants(
                farmgymobs_to_obs(obs1), id_of_plants_stage=7
            ), {}

    def step(self, action):
        # Stepping
        #   farmgym run with a cycle of 2 steps: 1 (empty) step of getting observation ("morning"), then 1 step of acting ("afternoon").
        #   Classic RL methodology use only 1 step : performing an action return the next observation
        #   To match this 2, we run the 'farmgy observation step ("morning")' right after the action.
        #   With this method, it will be like classic RL 'step' for the user
        _, reward, is_done, info = self.farm.farmgym_step(self.num_to_action(action))
        #_, reward, is_done, info = self.farm.farmgym_step(self.farm.gymaction_to_discretized_farmgymaction([action]))
        obs1, _, _, info = self.farm.gym_step([])

        observation = observation_hide_final_state_of_plants( farmgymobs_to_obs(obs1), id_of_plants_stage=7)

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
        else:
            return []  # Do nothing.


