import os
import matplotlib.pyplot as plt
import random
from typing import Dict, Any
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from lib.config import LOG_DIR
from lib.progress_callback import ProgressCallback


def create_dir(directory: str) -> None:
    """
    Creates a directory if it does not exist.

    Args:
        directory (str): Path of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_ppo_metrics(configuration: str, algo: str, callback: ProgressCallback, log_dir: str):
    """
    Produce PPO-specific plots: 
    - returns
    - policy gradient loss
    - value loss
    - entropy loss
    - approx_kl
    - etc.
    """
    path = os.path.join(log_dir, algo, configuration, "train")
    os.makedirs(path, exist_ok=True)

    # 1. Plot Returns
    plt.figure(figsize=(10, 5))
    plt.plot(callback.returns, label="Episode Reward")
    plt.title("Episode Rewards Over Time (PPO)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/ppo_returns.png")
    plt.close()

    # 2. Plot PPO Losses in one figure
    plt.figure(figsize=(10, 5))
    # 'loss' is the total loss in PPO (key: 'loss')
    # 'policy_gradient_loss' is also relevant in PPO
    if "loss" in callback.metric_history:
        plt.plot(callback.metric_history["loss"], label="Total Loss")
    if "policy_gradient_loss" in callback.metric_history:
        plt.plot(callback.metric_history["policy_gradient_loss"], label="Policy Gradient Loss")
    if "value_loss" in callback.metric_history:
        plt.plot(callback.metric_history["value_loss"], label="Value Loss")
    plt.title("PPO Losses Over Time")
    plt.xlabel("Training Updates")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/ppo_losses.png")
    plt.close()

    # 3. Plot Additional PPO metrics: entropy_loss, approx_kl, explained_variance, etc.
    plt.figure(figsize=(10, 5))
    if "entropy_loss" in callback.metric_history:
        plt.plot(callback.metric_history["entropy_loss"], label="Entropy Loss")
    if "approx_kl" in callback.metric_history:
        plt.plot(callback.metric_history["approx_kl"], label="Approx KL")
    if "explained_variance" in callback.metric_history:
        plt.plot(callback.metric_history["explained_variance"], label="Explained Variance")
    plt.title("PPO Auxiliary Metrics")
    plt.xlabel("Training Updates")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/ppo_auxiliary_metrics.png")
    plt.close()


def plot_a2c_metrics(configuration: str, algo: str, callback: ProgressCallback, log_dir: str):
    """
    Produce A2C-specific plots:
    - returns
    - policy_loss
    - value_loss
    - entropy_loss
    - explained_variance
    """
    path = os.path.join(log_dir, algo, configuration, "train")
    os.makedirs(path, exist_ok=True)

    # 1. Returns
    plt.figure(figsize=(10, 5))
    plt.plot(callback.returns, label="Episode Reward")
    plt.title("Episode Rewards Over Time (A2C)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/a2c_returns.png")
    plt.close()

    # 2. Losses
    plt.figure(figsize=(10, 5))
    if "policy_loss" in callback.metric_history:
        plt.plot(callback.metric_history["policy_loss"], label="Policy Loss")
    if "value_loss" in callback.metric_history:
        plt.plot(callback.metric_history["value_loss"], label="Value Loss")
    plt.title("A2C Losses Over Time")
    plt.xlabel("Training Updates")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/a2c_losses.png")
    plt.close()

    # 3. Entropy, explained variance, etc.
    plt.figure(figsize=(10, 5))
    if "entropy_loss" in callback.metric_history:
        plt.plot(callback.metric_history["entropy_loss"], label="Entropy Loss")
    if "explained_variance" in callback.metric_history:
        plt.plot(callback.metric_history["explained_variance"], label="Explained Variance")
    plt.title("A2C Auxiliary Metrics")
    plt.xlabel("Training Updates")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/a2c_auxiliary_metrics.png")
    plt.close()


def plot_td3_metrics(configuration: str, algo: str, callback: ProgressCallback, log_dir: str):
    """
    Produce TD3-specific plots:
    - returns
    - actor_loss
    - critic_loss
    - (optionally) learning_rate, etc.
    """
    path = os.path.join(log_dir, algo, configuration, "train")
    os.makedirs(path, exist_ok=True)

    # 1. Returns
    plt.figure(figsize=(10, 5))
    plt.plot(callback.returns, label="Episode Reward")
    plt.title("Episode Rewards Over Time (TD3)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/td3_returns.png")
    plt.close()

    # 2. Actor/Critic Loss
    plt.figure(figsize=(10, 5))
    if "actor_loss" in callback.metric_history:
        plt.plot(callback.metric_history["actor_loss"], label="Actor Loss")
    if "critic_loss" in callback.metric_history:
        plt.plot(callback.metric_history["critic_loss"], label="Critic Loss")
    plt.title("TD3 Actor/Critic Loss")
    plt.xlabel("Training Updates")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/td3_losses.png")
    plt.close()

    # 3. Extra metrics (learning_rate, etc.)
    plt.figure(figsize=(10, 5))
    if "learning_rate" in callback.metric_history:
        plt.plot(callback.metric_history["learning_rate"], label="Learning Rate")
    if "n_updates" in callback.metric_history:
        plt.plot(callback.metric_history["n_updates"], label="Number of Updates")
    plt.title("TD3 Auxiliary Metrics")
    plt.xlabel("Training Updates")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/td3_auxiliary_metrics.png")
    plt.close()


def random_search(
    algo_class,
    env_id: str,
    hyperparameter_ranges: Dict[str, list],
    n_iterations: int = 10,
    training_timesteps: int = 100_000,
    eval_episodes: int = 10
) -> Dict[str, Any]:
    """
    Performs random search to find the best hyperparameters for a given algorithm class.

    Args:
        algo_class: The agent class (PPOAgent, SACAgent, A2CAgent, etc.).
        env_id (str): ID of the Gym environment.
        hyperparameter_ranges (Dict[str, list]): Hyperparameter search space.
        n_iterations (int, optional): Number of random iterations. Defaults to 10.
        training_timesteps (int, optional): Timesteps for training each model. Defaults to 100_000.
        eval_episodes (int, optional): Number of episodes for evaluation. Defaults to 10.

    Returns:
        Dict[str, Any]: Best hyperparameters found.
    """
    best_score = -float('inf')
    best_params = None

    # Create a separate evaluation environment
    eval_env = make_vec_env(env_id, n_envs=1)

    for i in range(n_iterations):
        params = {key: random.choice(values) for key, values in hyperparameter_ranges.items()}
        print(f"Iteration {i + 1}/{n_iterations} with params: {params}")

        # Create a new training environment for each iteration
        train_env = make_vec_env(env_id, n_envs=4)

        # Use the agent class’s logic
        agent = algo_class(env_id, params, verbose=0)
        agent.create_model()
        agent.model.learn(total_timesteps=training_timesteps)

        mean_reward, std_reward = evaluate_policy(agent.model, eval_env, n_eval_episodes=eval_episodes, warn=False)
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward}\n")

        if mean_reward > best_score:
            best_score = mean_reward
            best_params = params

        # Close the training environment to free resources
        train_env.close()

    # Close the evaluation environment
    eval_env.close()

    print(f"Best parameters: {best_params}")
    print(f"Best reward: {best_score}")

    # Save the best parameters to a file
    agent_module_name = algo_class.__module__.split('.')[-1] 
    save_path = f"{LOG_DIR}/{agent_module_name}_best_params.txt"
    create_dir(f"{LOG_DIR}/")
    with open(save_path, 'w') as f:
        f.write(f"Best Reward: {best_score}\n")
        f.write(f"Best Parameters: {best_params}\n")
    print(f"Best parameters saved to {save_path}\n")

    return best_params
