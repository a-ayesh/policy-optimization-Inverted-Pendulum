import os
import matplotlib.pyplot as plt
import random
from typing import Dict, Any
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from lib.config import LOG_DIR


def create_dir(directory: str) -> None:
    """
    Creates a directory if it does not exist.

    Args:
        directory (str): Path of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_returns(configuration: str, algo: str, callback: Any) -> None:
    """
    Plots episode rewards over time.
    """
    path = f"{LOG_DIR}/{algo}/{configuration}/train"
    create_dir(path)

    plt.figure(figsize=(12, 6))
    plt.plot(callback.returns, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/training_returns.png")
    plt.close()


def plot_training_losses(configuration: str, algo: str, callback: Any) -> None:
    """
    Plots network losses over time.
    """
    path = f"{LOG_DIR}/{algo}/{configuration}/train"
    create_dir(path)

    plt.figure(figsize=(12, 6))
    plt.plot(callback.losses, label='Total Loss', color='red')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.title('Total Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/training_losses.png")
    plt.close()


def plot_value_deltas(configuration: str, algo: str, callback: Any) -> None:
    """
    Plots the delta between Monte Carlo estimate and actual value function.
    """
    path = f"{LOG_DIR}/{algo}/{configuration}/train"
    create_dir(path)

    plt.figure(figsize=(12, 6))
    plt.plot(callback.value_losses, label='Value Loss Delta', color='orange')
    plt.xlabel('Update Step')
    plt.ylabel('Delta')
    plt.title('Delta in Value Estimates Across Updates')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/training_value_deltas.png")
    plt.close()


def plot_policy_losses(configuration: str, algo: str, callback: Any) -> None:
    """
    Plots policy losses over time.
    """
    path = f"{LOG_DIR}/{algo}/{configuration}/train"
    create_dir(path)

    plt.figure(figsize=(12, 6))
    plt.plot(callback.policy_losses, label='Policy Gradient Loss', color='green')
    plt.xlabel('Update Step')
    plt.ylabel('Policy Loss')
    plt.title('Policy Gradient Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/training_policy_losses.png")
    plt.close()


def plot_entropy_losses(configuration: str, algo: str, callback: Any) -> None:
    """
    Plots entropy losses over time.
    """
    path = f"{LOG_DIR}/{algo}/{configuration}/train"
    create_dir(path)

    if not callback.entropy_losses:
        print(f"No entropy losses to plot for '{configuration}' configuration.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(callback.entropy_losses, label='Entropy Loss', color='purple')
    plt.xlabel('Update Step')
    plt.ylabel('Entropy')
    plt.title('Entropy Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/training_entropy_losses.png")
    plt.close()


def plot_additional_metrics(configuration: str, algo: str, callback: Any) -> None:
    """
    Plots additional training metrics over time.
    """
    path = f"{LOG_DIR}/{algo}/{configuration}/train"
    create_dir(path)

    metrics = {
        'Approx KL': callback.approx_kl,
        'Explained Variance': callback.explained_variance,
        'Standard Deviation': callback.std
    }

    for metric_name, data in metrics.items():
        if not data:
            print(f"No data for '{metric_name}' to plot in '{configuration}' configuration.")
            continue

        plt.figure(figsize=(12, 6))
        plt.plot(data, label=metric_name)
        plt.xlabel('Update Step')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{path}/training_{metric_name.lower().replace(' ', '_')}.png")
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

    return best_params
