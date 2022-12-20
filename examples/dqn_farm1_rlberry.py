"""
rlberry PPO on Farm1
====================
"""


import farmgym_games
import numpy as np
from rlberry.envs import gym_make

from rlberry.agents.torch import DQNAgent
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
from rlberry.agents.torch.utils.training import model_factory_from_env
import numpy as np

from rlberry.agents.torch.utils.training import model_factory_from_env
import numpy as np
# import logging
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)


policy_configs = {
    "type": "MultiLayerPerceptron",  # A network architecture
    "layer_sizes": (256, 256),  # Network dimensions
    "reshape": False,
    "is_policy": True,
}

value_configs = {
    "type": "MultiLayerPerceptron",
    "layer_sizes": (256, 256),
    "reshape": False,
    "out_size": 1,
}


env_ctor, env_kwargs = gym_make, {"id": "Farm1-v0"}

if __name__ == "__main__":
    manager = AgentManager(
        DQNAgent,
        (env_ctor, env_kwargs),
        agent_name="DQNAgent",
        init_kwargs=dict(
            learning_rate=9e-5,
        ),
        fit_budget=5e5,
        eval_kwargs=dict(eval_horizon=365),
        n_fit=1,
        parallelization="process",
        mp_context="spawn",
        enable_tensorboard=True,
    )
    manager.fit()
    eval_means = []
    for id_agent in range(20):
        eval_means.append(
                np.mean(manager.eval_agents(100, agent_id=id_agent))
            )
    print(eval_means)
