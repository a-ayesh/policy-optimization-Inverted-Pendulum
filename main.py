import argparse

from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from agents.a2c_agent import A2CAgent

from lib.progress_callback import ProgressCallback
from lib.config import (
    TRAINING_TIMESTEPS,
    N_EVAL_EPISODES,
    ENV_ID,
    LOG_DIR,

    # PPO
    PPO_DEFAULT_HYPERPARAMS,
    PPO_OPTIMIZED_HYPERPARAMS,
    PPO_HYPERPARAMETER_RANGES,

    # SAC
    SAC_DEFAULT_HYPERPARAMS,
    SAC_OPTIMIZED_HYPERPARAMS,
    SAC_HYPERPARAMETER_RANGES,

    # A2C
    A2C_DEFAULT_HYPERPARAMS,
    A2C_OPTIMIZED_HYPERPARAMS,
    A2C_HYPERPARAMETER_RANGES,
)

from lib.utils import *


def select_agent_class(algo_name: str):
    """
    Return the Agent class corresponding to the algorithm name.
    """
    if algo_name.lower() == 'ppo':
        return PPOAgent
    elif algo_name.lower() == 'sac':
        return SACAgent
    elif algo_name.lower() == 'a2c':
        return A2CAgent
    else:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Supported: ppo, sac, a2c.")


def get_default_hyperparams(algo_name: str):
    """
    Return default hyperparams for the specified algorithm.
    """
    if algo_name.lower() == 'ppo':
        return PPO_DEFAULT_HYPERPARAMS
    elif algo_name.lower() == 'sac':
        return SAC_DEFAULT_HYPERPARAMS
    elif algo_name.lower() == 'a2c':
        return A2C_DEFAULT_HYPERPARAMS
    else:
        raise ValueError(f"Unknown algorithm '{algo_name}'.")


def get_optimized_hyperparams(algo_name: str):
    """
    Return the optimized hyperparams dict for the specified algorithm.
    (We'll update this dict in-place after random search.)
    """
    if algo_name.lower() == 'ppo':
        return PPO_OPTIMIZED_HYPERPARAMS
    elif algo_name.lower() == 'sac':
        return SAC_OPTIMIZED_HYPERPARAMS
    elif algo_name.lower() == 'a2c':
        return A2C_OPTIMIZED_HYPERPARAMS
    else:
        raise ValueError(f"Unknown algorithm '{algo_name}'.")


def get_hyperparameter_ranges(algo_name: str):
    """
    Return the hyperparameter ranges for random search for the specified algorithm.
    """
    if algo_name.lower() == 'ppo':
        return PPO_HYPERPARAMETER_RANGES
    elif algo_name.lower() == 'sac':
        return SAC_HYPERPARAMETER_RANGES
    elif algo_name.lower() == 'a2c':
        return A2C_HYPERPARAMETER_RANGES
    else:
        raise ValueError(f"Unknown algorithm '{algo_name}'.")


def train_and_evaluate(agent, algo, callback, total_timesteps, configuration_label):
    """
    Trains the agent, evaluates it, and plots the training metrics.
    """
    print(f"Training '{algo}' agent with '{configuration_label}' hyperparameters...")
    agent.train(total_timesteps=total_timesteps, callback=callback)
    print("Training complete.\n")

    print(f"Evaluating '{algo}' agent with '{configuration_label}' hyperparameters...")
    mean_reward, std_reward = agent.evaluate(n_eval_episodes=N_EVAL_EPISODES)
    print(f"Stats after training the agent ({configuration_label}):")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward}\n")

    print(f"Plotting training metrics for '{configuration_label}' configuration...")
    plot_returns(configuration_label, algo, callback)
    plot_training_losses(configuration_label, algo, callback)
    plot_value_deltas(configuration_label, algo, callback)
    plot_policy_losses(configuration_label, algo, callback)
    plot_entropy_losses(configuration_label, algo, callback)
    plot_additional_metrics(configuration_label, algo, callback)
    print(f"Plots saved for '{configuration_label}' configuration.\n")


def run_for_algorithm(algo_name: str, do_search: bool):
    """
    Executes the training/evaluation workflow for a single algorithm.
    """
    print(f"Selected algorithm: {algo_name.upper()}")

    # Get relevant hyperparams and agent class
    agent_class = select_agent_class(algo_name)
    default_hparams = get_default_hyperparams(algo_name)
    optimized_hparams = get_optimized_hyperparams(algo_name)
    hyperparam_ranges = get_hyperparameter_ranges(algo_name)

    # ==============
    # Train with Default Params
    # ==============
    print(f"=== Training {algo_name.upper()} with Default Hyperparameters ===\n")
    agent_default = agent_class(env_id=ENV_ID, hyperparams=default_hparams, verbose=1)
    agent_default.create_model()

    # Evaluate before training (Random Agent)
    print("Evaluating before training (Default Params)...")
    mean_reward, std_reward = agent_default.evaluate(n_eval_episodes=N_EVAL_EPISODES)
    print(f"Stats before training (Default Params):\nMean Reward: {mean_reward:.2f} +/- {std_reward}\n")

    # Instantiate the callback
    callback_default = ProgressCallback()

    # Train and evaluate
    train_and_evaluate(agent_default, algo_name, callback_default, TRAINING_TIMESTEPS, 'default')

    # ==============
    # Hyperparameter Search (optional)
    # ==============
    if do_search:
        print(f"=== Performing Random Hyperparameter Search for {algo_name.upper()} ===\n")
        best_params = random_search(
            algo_class=agent_class,
            env_id=ENV_ID,
            hyperparameter_ranges=hyperparam_ranges,
            n_iterations=25,                 # adjust as you wish
            training_timesteps=TRAINING_TIMESTEPS,
            eval_episodes=N_EVAL_EPISODES
        )
        optimized_hparams.update(best_params)  # update the global dictionary
        print(f"Best Hyperparameters Found: {best_params}\n")

        # ==============
        # Train with Optimized Params
        # ==============
        print(f"=== Training {algo_name.upper()} with Optimized Hyperparameters ===\n")
        agent_optimized = agent_class(env_id=ENV_ID, hyperparams=optimized_hparams, verbose=1)
        agent_optimized.create_model()

        # Evaluate before training with optimized params (Random Agent)
        print("Evaluating before training (Optimized Params)...")
        mean_reward, std_reward = agent_optimized.evaluate(n_eval_episodes=N_EVAL_EPISODES)
        print(f"Stats before training (Optimized Params):\nMean Reward: {mean_reward:.2f} +/- {std_reward}\n")

        # Instantiate a new callback for optimized run
        callback_optimized = ProgressCallback()
        train_and_evaluate(agent_optimized, algo_name, callback_optimized, TRAINING_TIMESTEPS, 'optimized')
    else:
        print(f"=== Hyperparameter Search Skipped for {algo_name.upper()} ===\n")

    # Final Cleanup
    print(f"Closing {algo_name.upper()} environment...")
    agent_default.env.close()
    if do_search:
        agent_optimized.env.close()
    print(f"{algo_name.upper()} environment closed.\n")


def main():
    parser = argparse.ArgumentParser(description="RL Project Main Script")
    parser.add_argument('--algo', type=str, default='ppo',
                        help='Algorithm to use: ppo, sac, a2c, or all. Default is ppo.')
    parser.add_argument('--search', action='store_true',
                        help='Perform hyperparameter search before training with optimized parameters.')
    args = parser.parse_args()

    algo_name = args.algo.lower()

    if algo_name == 'all':
        # Run workflow for PPO, then SAC, then A2C
        for alg in ['ppo', 'sac', 'a2c']:
            run_for_algorithm(alg, args.search)
    else:
        # Run workflow for the selected algorithm
        run_for_algorithm(algo_name, args.search)


if __name__ == "__main__":
    main()
