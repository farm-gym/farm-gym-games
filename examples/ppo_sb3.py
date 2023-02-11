import farmgym_games
import gym
from stable_baselines3 import PPO


env = gym.make("OldV21Farm1-v0", render_mode="human") # compatibility version

if __name__ == "__main__":
    model = PPO("MlpPolicy", env, verbose=1)
    # Training
    model.learn(total_timesteps=10)
    # Evaluation

    obs = env.reset()
    ep_rew = 0
    n_episodes = 0
    while True:
       action, _states = model.predict(obs, deterministic=True)
       obs, reward, done, info = env.step(action)
       ep_rew += reward
       env.render()
       if done:
           obs = env.reset()
           print(f'Episode Reward on evaluation was {ep_rew}')
           break
    env.close()
