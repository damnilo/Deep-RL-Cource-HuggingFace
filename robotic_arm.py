import os

import gymnasium as gym
import panda_gym

from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from huggingface_hub import notebook_login

env_id = "PandaReachDense-v3"

# env = gym.make(env_id)

# s_size = env.observation_space.shape
# a_size = env.action_space

# print("OBSERVATION SPACE: ", s_size)
# print("Obs sample: ", env.observation_space.sample())

# print("ACTION SPACE: ", a_size)
# print("Action sample: ", env.action_space.sample())

# env = make_vec_env(env_id, n_envs=4)
# env = VecNormalize(venv=env, norm_obs=True, norm_reward=True, clip_obs=10.)

# model = A2C(policy="MultiInputPolicy", env=env, verbose=1)

# model.learn(500_000)

# model.save("a2c-PandaReachDense-v3")
# env.save("vec_normalize.pk1")

eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
eval_env = VecNormalize(eval_env)

eval_env.render_mode = "rgb_array"

eval_env.training = False
eval_env.norm_reward = False

model = A2C.load("a2c-PandaReachDense-v3")

mean_reward, std_reward = evaluate_policy(model, eval_env)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")