import os
import time
import numpy as np
from farmgym.v2.farm import generate_video#, generate_gif

def run_gym_xp(farm, agent, max_steps=np.infty, render=True, monitoring=False):
    agent.reset(farm)
    observation, information = farm.reset()
    if render == "text":
        print("Initial step:")
        print(farm.render_step([], observation, 0, False, False, information))
        print("###################################")
    elif render == "image":
        time_tag = time.time()
        os.mkdir("run-" + str(time_tag))
        os.chdir("run-" + str(time_tag))
        farm.render()
    agent.init(observation)

    terminated = False
    i = 0
    while (not terminated) and i <= max_steps:

        action = agent.choose_action()
        obs, reward, terminated, truncated, info = farm.step(action)
        if render == "text":
            print(farm.render_step(action, obs, reward, terminated, truncated, info))
            print("###################################")
        elif render == "image":
            farm.render()
        agent.update(obs, reward, terminated, truncated, info)
        i += 1
    if render == "image":
        farm.render()
        generate_video(image_folder=".", video_name="farm.avi")
        os.chdir("../")