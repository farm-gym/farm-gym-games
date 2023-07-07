import os
from pathlib import Path

import numpy as np
from gymnasium.envs.registration import register

file_path = Path(os.path.realpath(__file__))
CURRENT_DIR = file_path.parent


def register_farms():
    environments = []

    def make_game_ids(directory_path, prefix=""):
        dirs = os.listdir(directory_path)
        farms = []
        for xx in dirs:
            if ".yaml" in xx:
                if ("actions" not in xx) and ("init" not in xx) and ("score" not in xx):
                    farms.append((xx, directory_path))

            if "." not in xx and xx[0] not in ["_"]:
                [
                    farms.append((y, p))
                    for y, p in make_game_ids(
                        os.path.join(directory_path, xx),
                        prefix + "/" + xx if prefix != "" else xx,
                    )
                ]
        return farms

    game_ids = make_game_ids(CURRENT_DIR)

    for x, path in game_ids:
        xx = x[:-5]
        # print("register", xx, "at ", path+"/"+xx +".yaml")
        register(
            id=xx + "-v0",
            entry_point="farmgym.v2.games.make_farm:make_farm",
            max_episode_steps=np.infty,
            reward_threshold=np.infty,
            kwargs={"yamlfile": path + "/" + xx + ".yaml"},
        )
        environments.append(xx + "-v0")
    return environments


if __name__ == "__main__":
    env_names = register_farms()
    print("List of FarmGym environments:")
    for env_name in env_names:
        print(env_name)

    # env_names = register_all()
    # print("List of FarmGym environments:")
    # for env_name in env_names:
    #    print(env_name)

    # from farmgym.v2.games.rungame import run_randomactions

    # env = gym.make(env_names[1])

    # farm = env.unwrapped
    ##understand_the_farm(farm)
    # Run some example:
    # run_randomactions(farm, max_steps=100, render=False, monitoring=True)
