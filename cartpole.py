import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
import gym_pygame

from huggingface_hub import notebook_login
import imageio

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import json

import tempfile

import os

env_id = "CartPole-v1"

env = gym.make(env_id, render_mode="rgb_array")
eval_env = gym.make(env_id, render_mode="rgb_array")

s_size = int(env.observation_space.shape[0])
a_size = int(env.action_space.n)

print("OBSERVATION SPACE:", s_size)
print("OBSERVATION SPACE:", env.observation_space.sample())
print("ACTION SPACE:", a_size)
print("Sample observation:", env.action_space.sample())

cartpole_hyperparams = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size
}

class Policy(nn.Module):

    def __init__(self, s_size, a_size, h_size):

        super(Policy, self).__init__()

        self.fc1 = nn.Linear(s_size, h_size)

        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim=1)

    def act(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)

def reinforce(policy, optimizer, episodes, max_t, gamma, print_every):

    scores_deque = deque(maxlen=100)
    scores = []

    for i in range(1, episodes + 1):

        saved_log_probs = []
        rewards = []
        state = env.reset()[0]

        for t in range(max_t):

            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        for t in range(n_steps):

            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft(rewards[t] + gamma * disc_return_t)

        eps = np.finfo(np.float32).eps.item()

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):

            policy_loss.append(-log_prob * disc_return)

        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i % print_every == 0:
            print('Episodr {}\tAverage Score: {:2f}'.format(i, np.mean(scores_deque)))

    return scores

def evaluate(env, max_steps, eval_episodes, policy):

    episode_rewards = []
    for episode in range(eval_episodes):

        state, _ = env.reset()
        step = 0
        done = False
        total_reward_ep = 0

        for step in range(max_steps):

            action, _ = policy.act(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward_ep += reward

            if terminated or truncated:
                break

            state = new_state

        episode_rewards.append(total_reward_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def record_video(env, policy, out_directory, fps=30):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    term = False
    trunc = False
    state, _ = env.reset()
    img = env.render()
    images.append(img)
    while not term and not trunc:
        # Take the action (index) that have the maximum expected future reward given that state
        action, _ = policy.act(state)
        state, reward, term, trunc, info = env.step(action)  # We directly put next_state = state for recording logic
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)

def push_to_hub(repo_id,
                model,
                hyperparameters,
                eval_env,
                video_fps=30
                ):
  """
  Evaluate, Generate a video and Upload a model to Hugging Face Hub.
  This method does the complete pipeline:
  - It evaluates the model
  - It generates the model card
  - It generates a replay video of the agent
  - It pushes everything to the Hub

  :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
  :param model: the pytorch model we want to save
  :param hyperparameters: training hyperparameters
  :param eval_env: evaluation environment
  :param video_fps: how many frame per seconds to record our video replay
  """

  _, repo_name = repo_id.split("/")
  api = HfApi()

  # Step 1: Create the repo
  repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
  )

  with tempfile.TemporaryDirectory() as tmpdirname:
    local_directory = Path(tmpdirname)

    # Step 2: Save the model
    torch.save(model, local_directory / "model.pt")

    # Step 3: Save the hyperparameters to JSON
    with open(local_directory / "hyperparameters.json", "w") as outfile:
      json.dump(hyperparameters, outfile)

    # Step 4: Evaluate the model and build JSON
    mean_reward, std_reward = evaluate(eval_env,
                                            hyperparameters["max_t"],
                                            hyperparameters["n_evaluation_episodes"],
                                            model)
    # Get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
          "env_id": hyperparameters["env_id"],
          "mean_reward": mean_reward,
          "n_evaluation_episodes": hyperparameters["n_evaluation_episodes"],
          "eval_datetime": eval_form_datetime,
    }

    # Write a JSON file
    with open(local_directory / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 5: Create the model card
    env_name = hyperparameters["env_id"]

    metadata = {}
    metadata["tags"] = [
          env_name,
          "reinforce",
          "reinforcement-learning",
          "custom-implementation",
          "deep-rl-class"
      ]

    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_name,
        dataset_id=env_name,
      )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    model_card = f"""
  # **Reinforce** Agent playing **{env_id}**
  This is a trained model of a **Reinforce** agent playing **{env_id}** .
  To learn to use this model and train yours check Unit 4 of the Deep Reinforcement Learning Course: https://huggingface.co/deep-rl-course/unit4/introduction
  """

    readme_path = local_directory / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
          readme = f.read()
    else:
      readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
      f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    # Step 6: Record a video
    video_path =  local_directory / "replay.mp4"
    record_video(env, model, video_path, video_fps)

    # Step 7. Push everything to the Hub
    api.upload_folder(
          repo_id=repo_id,
          folder_path=local_directory,
          path_in_repo=".",
    )

    print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")

debug_policy = Policy(s_size, a_size, 64)
debug_policy.act(env.reset()[0])

cartpole_policy = Policy(
    cartpole_hyperparams["state_space"],
    cartpole_hyperparams["action_space"],
    cartpole_hyperparams["h_size"]
)
cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparams["lr"])

scores = reinforce(
    cartpole_policy,
    cartpole_optimizer,
    cartpole_hyperparams["n_training_episodes"],
    cartpole_hyperparams["max_t"],
    cartpole_hyperparams["gamma"],
    100
)

push_to_hub(
    "damnilo/cartpole-v1", 
    cartpole_policy, 
    cartpole_hyperparams,
    eval_env,
    video_fps=30 
)