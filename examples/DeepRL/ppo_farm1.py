"""
PPO on Farm1
============
"""

from stable_baselines3 import PPO

from  farmgym_games import Farm1
import numpy as np

if __name__ == "__main__":
    env = Farm1()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    obs = env.reset()
    cum_rew = 0
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        cum_rew += reward
        if done : 
            break
    print("Reward is ", cum_rew)

