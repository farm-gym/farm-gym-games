import numpy as np
import pytest
from gymnasium.spaces import Discrete

from farmgym_games.game_agents.basic_agents import Farmgym_RandomAgent
from farmgym_games.game_builder.check_gym_env import check_gym_env
from farmgym_games.game_catalogue.farm0.farm import env as Farm0
from farmgym_games.game_catalogue.farm1.farm import env as Farm1
from farmgym_games.game_catalogue.farm2.farm import env as Farm2

ALL_ENVS = [Farm0, Farm1, Farm2]

@pytest.mark.parametrize("Env", ALL_ENVS)
def test_env(Env):
    """
    Check that the environment is (almost) gym-compatible
    """
    check_gym_env(Env())

@pytest.mark.skip(reason="For some reason farms are not reproducible")
@pytest.mark.parametrize("Env", ALL_ENVS)
def test_reproducibility(Env):
    """
    Check if the environment is reproducible in the sense that
    it returns the same states when given the same seed.
    """
    env = Env()
    action = env.action_space.sample()
    env.reset(seed=42)
    a = env.step(action)[0]

    env.reset(seed=42)
    b = env.step(action)[0]
    if hasattr(a, "__len__"):
        assert np.all(
            np.array(a) == np.array(b)
        ), "The environment does not seem to be reproducible"
    else:
        assert a == b, "The environment does not seem to be reproducible"


@pytest.mark.parametrize("Env", ALL_ENVS)
def test_env_interaction(Env):
    """
    Tests the environment by running a Farmgym_RandomAgent for 10 steps
    """
    actions = []
    env = Env()
    agent = Farmgym_RandomAgent()
    agent.reset(env)
    for i in range(10):
        action = agent.choose_action()
        observation, reward, done, _, _ = agent.farm.step(action)
        actions.append(action)
    assert actions

def test_farmgym_random_agent():
    """
    Tests Farmgym_RandomAgent for 10 steps in a fake farm
    """ 
    class FakeFarm:
        def __init__(self):
            self.action_space = Discrete(2)
        def step(self, action):
            return np.zeros(2), 1, False, False, {}  

    actions = []
    env = FakeFarm()
    agent = Farmgym_RandomAgent()
    agent.reset(env)
    for i in range(10):
        action = agent.choose_action()
        observation, reward, done, _, _ = agent.farm.step(action)
        actions.append(action)
    assert actions
