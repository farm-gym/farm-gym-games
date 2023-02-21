"""
Expert agent on Farm0
=====================
"""

from rlberry.agents import AgentWithSimplePolicy
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
from rlberry.envs import gym_make
import farmgym_games
import numpy as np

env_ctor, env_kwargs = gym_make, {"id": "OldV21Farm0-v0"}
starting_day_for_policy = 100


class ExpertAgent(AgentWithSimplePolicy):
    name = "ExpertAgentFarm1"
    fruit_stage_duration_count = 0
    previous_weight = 0

    def __init__(self, env, starting_day_for_policy=0, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.starting_day_for_policy = starting_day_for_policy

    def fit(self, budget=100, **kwargs):
        observation = self.env.reset()
        episode_reward = 0
        for ep in range(int(budget)):
            action = self.policy(observation)
            observation, reward, done, info = self.env.step(action)
            episode_reward += reward
            if done:
                self.writer.add_scalar("episode_rewards", episode_reward, ep)
                episode_reward = 0
                self.env.reset()

    def policy(self, observation):
        # Policy : TO EXPLAIN

        # The available actions are :
        #     0) Do nothing
        #     1) Pour 1L of water
        #     2) Pour 5L of water
        #     3) Harvest the plant
        #     4) sow
        #     5) Fertilizer
        #     6) Herbicide
        #     7) Pesticide
        #     8) Remove weeds by hand

        # observation[0] : Day (from 1 to 365)
        # observation[1] : Mean air temperature (°C)
        # observation[2] : Min air temperature (°C)
        # observation[3] : Max air temperature (°C)
        # observation[4] : Rain amount
        # observation[5] : Sun-exposure (from 1 to 5)
        # observation[6] : Consecutive dry day (int)
        # observation[7] : Stage of growth of the plant
        # observation[8] : Number of fruits (int)
        # observation[9] : Size of the plant in cm
        # observation[10] : Soil wet_surface (m2.day-1)
        # observation[11] : fertilizer amount (kg)
        # observation[12] : Pollinators occurrence (bin)
        # observation[13] : Weeds grow (nb)
        # observation[14] : Weeds flowers (nb)
        # observation[15] : weight of fruits
        # observation[16] : microlife health index (%)

        # states for 'stage for plants' (observation[7]) are :
        # 0:'none'
        # 1:'seed'
        # 2:'entered_grow'
        # 3:'grow'
        # 4:'entered_bloom'
        # 5:'bloom'
        # 6:'entered_fruit'
        # 7:'fruit'
        # 8:'entered_ripe'
        # 9:'ripe'
        # 10:'entered_seed'
        # 11:'harvested'
        # 12:'dead'

        # print(self.env.farm.get_free_observations())

        next_action = 0  # default
        if observation[0] == starting_day_for_policy:
            next_action = 2  # 5L of water
        elif observation[0] == starting_day_for_policy + 1:
            next_action = 6  # herbicide
        elif observation[0] == starting_day_for_policy + 2:
            next_action = 7  # pesticide
        elif observation[0] == starting_day_for_policy + 3:
            next_action = 5  # Fertilizer
        elif observation[0] == starting_day_for_policy + 4:
            next_action = 4  # sow
        elif observation[0] == starting_day_for_policy + 5:
            next_action = 1  # 1L of water
        elif observation[0] > starting_day_for_policy + 5:
            if observation[7] in [6, 7, 8, 9]:
                if self.fruit_stage_duration_count > 4:
                    next_action = 3  # harvesting
                    self.fruit_stage_duration_count = 0
                else:
                    self.fruit_stage_duration_count += 1

        return next_action


class Agent(ExpertAgent):
    def __init__(self, env, **kwargs):
        ExpertAgent.__init__(self, env, starting_day_for_policy, **kwargs)


if __name__ == "__main__":
    manager = AgentManager(
        Agent,
        (env_ctor, env_kwargs),
        agent_name="ExpertAgentFarm1",
        fit_budget=1,
        eval_kwargs=dict(eval_horizon=365),
        n_fit=4,
        parallelization="process",
        mp_context="spawn",
        output_dir="expert_farm1_results",
    )
    manager.fit()
    evaluation = evaluate_agents([manager], n_simulations=128, plot=False)
    np.savetxt("expert_farm1.out", np.array(evaluation.values), delimiter=",")
    print(evaluation.describe())


