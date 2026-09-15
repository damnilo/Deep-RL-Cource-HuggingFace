import numpy as np
import gymnasium as gym
import random
import imageio
import os

import pickle
from tqdm import tqdm

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import json

episodes = 10000
lr = 0.7
eval_episodes = 100

env_id = "FrozenLake-v1"
max_steps = 99
gamma = 0.95
eval_seed = []

max_eps = 1.0
min_eps = 0.05
decay_rate = 0.0005

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")

state_space = env.observation_space.n

action_space = env.action_space.n

def initialize_q_table(state_space, action_space):

    return np.zeros((state_space, action_space))

def greedy_policy(q_table, state):

    return np.argmax(q_table[state][:])

def eps_greedy_policy(q_table, state, eps):

    random_num = random.uniform(0, 1)

    if random_num <= eps:

        return env.action_space.sample()

    else:

        return greedy_policy(q_table, state)

def train(episodes, max_steps, max_eps, min_eps, eps_decay, lr, env, q_table):

    for i in tqdm(range(episodes)):

        eps = max(max_eps - i * eps_decay, min_eps)
        state, _ = env.reset()
        step = 0
        term = False
        trunc = False

        for step in range(max_steps):

            action = eps_greedy_policy(q_table, state, eps)

            new_state, reward, term, trunc, _ = env.step(action)

            q_table[state][action] = q_table[state][action] + lr * (reward + gamma * max(q_table[new_state]) - q_table[state][action])

            if term or trunc:
                break

            state = new_state

    return q_table

def eval(env, max_steps, eval_episodes, q_table, seed):

    episode_rewards = []
    for episode in tqdm(range(eval_episodes)):

        if seed:

            state, _ = env.reset(seed=seed[episode])
        else:

            state, _ = env.reset()

        step = 0
        term = False
        trunc = False
        total_ep_reward = 0

        for step in range(max_steps):

            action = greedy_policy(q_table, state)
            new_state, reward, term, trunc, _ = env.step(action)
            total_ep_reward += reward

            if term or trunc:
                break

            state = new_state

        episode_rewards.append(total_ep_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward        

def record_video(Qtable, out_directory, fps=1):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    record_env = gym.make(
        "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array"
    )
    images = []
    terminated = False
    truncated = False
    state, info = record_env.reset(seed=random.randint(0, 500))
    img = record_env.render()
    images.append(img)
    while not terminated and not truncated:
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Qtable[state][:])
        state, reward, terminated, truncated, info = env.step(
            action
        )  # We directly put next_state = state for recording logic
        img = env.render()
        images.append(img)

    record_env.close()
    imageio.mimsave(out_directory, images, fps=fps)

def push_to_hub(repo_id, model, env, video_fps=1, local_repo_path="hub"):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the Hub

    :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
    :param env
    :param video_fps: how many frame per seconds to record our video replay
    (with taxi-v3 and frozenlake-v1 we use 1)
    :param local_repo_path: where the local repository is
    """
    _, repo_name = repo_id.split("/")

    eval_env = env
    api = HfApi()

    # Step 1: Create the repo
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
    )

    # Step 2: Download files
    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    # Step 3: Save the model
    if env.spec.kwargs.get("map_name"):
        model["map_name"] = env.spec.kwargs.get("map_name")
        if env.spec.kwargs.get("is_slippery", "") == False:
            model["slippery"] = False

    # Pickle the model
    with open((repo_local_path) / "q-learning.pkl", "wb") as f:
        pickle.dump(model, f)

    # Step 4: Evaluate the model and build JSON with evaluation metrics
    mean_reward, std_reward = eval(
        eval_env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"]
    )

    evaluate_data = {
        "env_id": model["env_id"],
        "mean_reward": mean_reward,
        "n_eval_episodes": model["n_eval_episodes"],
        "eval_datetime": datetime.datetime.now().isoformat(),
    }

    # Write a JSON file called "results.json" that will contain the
    # evaluation results
    with open(repo_local_path / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 5: Create the model card
    env_name = model["env_id"]
    if env.spec.kwargs.get("map_name"):
        env_name += "-" + env.spec.kwargs.get("map_name")

    if env.spec.kwargs.get("is_slippery", "") == False:
        env_name += "-" + "no_slippery"

    metadata = {}
    metadata["tags"] = [env_name, "q-learning", "reinforcement-learning", "custom-implementation"]

    # Add metrics
    evaluation = metadata_eval_result(
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
    metadata = {**metadata, **evaluation}

    model_card = f"""
  # **Q-Learning** Agent playing1 **{env_id}**
  This is a trained model of a **Q-Learning** agent playing **{env_id}** .

  ## Usage

  model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

  # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
  env = gym.make(model["env_id"])
  """

    eval(env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"])

    readme_path = repo_local_path / "README.md"
    readme = ""
    print(readme_path.exists())
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
    video_path = repo_local_path / "replay.mp4"
    record_video(model["qtable"], video_path, video_fps)

    # Step 7. Push everything to the Hub
    api.upload_folder(
        repo_id=repo_id,
        folder_path=repo_local_path,
        path_in_repo=".",
    )

    print("Your model is pushed to the Hub. You can view your model here: ", repo_url)

q_table_frozenlake = initialize_q_table(state_space, action_space)

q_table_frozenlake = train(episodes, max_steps, max_eps, min_eps, decay_rate, lr, env, q_table_frozenlake)

model = {
    "env_id": env_id,
    "max_steps": max_steps,
    "n_training_episodes": episodes,
    "n_eval_episodes": eval_episodes,
    "eval_seed": eval_seed,
    "learning_rate": lr,
    "gamma": gamma,
    "max_epsilon": max_eps,
    "min_epsilon": min_eps,
    "decay_rate": decay_rate,
    "qtable": q_table_frozenlake,
}

push_to_hub(
    repo_id="damnilo/frozenlake-q-learning",
    model=model,
    env=env,
    video_fps=1,
    local_repo_path="hub",
)

