import gymnasium

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import (
    notebook_login,
)  # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env = gymnasium.make("LunarLander-v3", render_mode="rgb_array")
env.reset()

print("OBSERVATION SPACE:", env.observation_space.shape)
print("Sample observation:", env.observation_space.sample())

print("ACTION SPACE:", env.action_space.shape)
print("Sample action:", env.action_space.sample())

env = make_vec_env("LunarLander-v3", n_envs=16)

model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98, ent_coef=0.01)

model.learn(total_timesteps=int(2e5))

eval_env = Monitor(gymnasium.make("LunarLander-v3", render_mode="rgb_array"))

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

package_to_hub(
    model=model,  # Our trained model
    model_name="ppo-LunarLander-v3",  # The name of our trained model
    model_architecture="PPO",  # The model architecture we used: in our case PPO
    env_id="LunarLander-v3",  # Name of the environment
    eval_env=eval_env,  # Evaluation Environment
    repo_id="damnilo/ppo-LunarLander-v3",  # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
    commit_message="Upload trained PPO model on LunarLander-v3",  # Commit message for the upload
)