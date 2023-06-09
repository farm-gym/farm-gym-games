import numpy as np
import pytest
from rlberry.agents import AgentWithSimplePolicy

from farmgym_games.game_catalogue.farm0.farm import env as Farm0
from farmgym_games.game_catalogue.farm1.farm import env as Farm1
from farmgym_games.game_catalogue.farm2.farm import env as Farm2
from farmgym_games.game_builder.check_gym_env import check_gym_env

ALL_ENVS = [Farm0, Farm1, Farm2]

@pytest.mark.parametrize("Env", ALL_ENVS)
def test_env(Env):
    """
    Check that the environment is (almost) gym-compatible
    """
    check_gym_env(Env())

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


class RandomAgent(AgentWithSimplePolicy):
    """
    Create a RandomAgent to test the environment
    """
    name = "RandomAgent"

    def __init__(self, env, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

    def fit(self, budget=100, **kwargs):
        actions = []
        observation = self.env.reset()
        episode_reward = 0
        for ep in range(int(budget)):
            action = self.policy(observation)
            observation, reward, done, _, _ = self.env.step(action)
            actions.append(action)
            episode_reward += reward
            if done:
                self.writer.add_scalar("episode_rewards", episode_reward, ep)
                episode_reward = 0
                self.env.reset()
        return actions

    def policy(self, observation):
        return self.env.action_space.sample()  # choose an action at random


@pytest.mark.parametrize("Env", ALL_ENVS)
def test_env_agent(Env):
    """
    Tests the environment by running a RandomAgent for 10 steps
    """
    env = Env()
    agent = RandomAgent(env)
    actions = agent.fit(10)
    assert actions