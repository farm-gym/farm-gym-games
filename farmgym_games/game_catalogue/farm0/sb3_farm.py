from stable_baselines3 import PPO
from farmgym_games.game_builder.utils_sb3 import farmgym_to_gym_observations_flattened, wrapper
from farmgym_games.game_catalogue.farm1.farm import env as Farm0

env = Farm0()
orignal_obs, _  = env.reset()
print(f"Original observation : \n{orignal_obs}\n")

# Wrap to change observation and action spaces and the step function
env.farmgym_to_gym_observations = farmgym_to_gym_observations_flattened
env = wrapper(env)
obs, _ = env.reset()
print(f"Wrapped observation : \n{obs}\n")

# Setup PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ppo_farmgym")

# Evaluate model

cr = 0
for i in range(10):
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, _, info = env.step(action)
        cr += rewards
        if dones:
            print("Reward = ",rewards)
            break


