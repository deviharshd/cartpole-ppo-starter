import gymnasium as gym
from stable_baselines3 import PPO

# 👇 add render_mode="human" so a window shows up
env = gym.make("CartPole-v1", render_mode="human")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=6000)

obs, info = env.reset()
for _ in range(500):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    # no need to call env.render() now, render_mode already does it
    if terminated or truncated:
        obs, info = env.reset()

env.close()
