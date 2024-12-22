import argparse
from agents.ppo_agent import PPOAgent
from lib.progress_callback import ProgressCallback
from lib.config import (
    TRAINING_TIMESTEPS,
    DEFAULT_HYPERPARAMS,
    OPTIMIZED_HYPERPARAMS,
    N_EVAL_EPISODES,
    ENV_ID,
    LOG_DIR,
    HYPERPARAMETER_RANGES
)
from lib.utils import *
from stable_baselines3.common.env_util import make_vec_env


def train_and_evaluate(agent, callback, total_timesteps, configuration_label):
    """
    Trains the agent, evaluates it, and plots the training metrics.

    Args:
        agent (BaseAgent): The RL agent to train.
        callback (BaseCallback): Callback for monitoring training.
        total_timesteps (int): Number of timesteps for training.
        configuration_label (str): Label for the training configuration.
    """
    print(f"Training agent with '{configuration_label}' hyperparameters...")
    agent.train(total_timesteps=total_timesteps, callback=callback)
    print("Training complete.\n")

    print(f"Evaluating agent with '{configuration_label}' hyperparameters...")
    mean_reward, std_reward = agent.evaluate(n_eval_episodes=N_EVAL_EPISODES)
    print(f"Stats after training the agent ({configuration_label}):")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward}\n")

    print(f"Plotting training metrics for '{configuration_label}' configuration...")
    plot_returns(configuration_label, callback)
    plot_training_losses(configuration_label, callback)
    plot_value_deltas(configuration_label, callback)
    plot_policy_losses(configuration_label, callback)
    plot_entropy_losses(configuration_label, callback)
    plot_additional_metrics(configuration_label, callback)
    print(f"Plots saved for '{configuration_label}' configuration.\n")


def main():
    """
    Main function to train and evaluate the PPO agent using default and optimized hyperparameters.
    """
    parser = argparse.ArgumentParser(description="RL Project Main Script")
    parser.add_argument('--search', action='store_true',
                        help='Perform hyperparameter search before training with optimized parameters.')
    args = parser.parse_args()

    # ============================
    # Step 1: Train with Default Params
    # ============================
    print("=== Training with Default Hyperparameters ===\n")
    ppo_default = PPOAgent(env_id=ENV_ID, hyperparams=DEFAULT_HYPERPARAMS, verbose=1)
    ppo_default.create_model()

    # Evaluate before training (Random Agent)
    print("Evaluating before training (Default Params)...")
    mean_reward, std_reward = ppo_default.evaluate(n_eval_episodes=N_EVAL_EPISODES)
    print(f"Stats before training (Default Params):\nMean Reward: {mean_reward:.2f} +/- {std_reward}\n")

    # Instantiate the callback
    callback = ProgressCallback()

    # Train and evaluate
    train_and_evaluate(ppo_default, callback, TRAINING_TIMESTEPS, 'default')

    # ============================
    # Step 2: Hyperparameter Search
    # ============================
    if args.search:
        print("=== Performing Random Hyperparameter Search ===\n")
        best_params = random_search(
            env_id=ENV_ID,
            hyperparameter_ranges=HYPERPARAMETER_RANGES,
            n_iterations=20,
            training_timesteps=TRAINING_TIMESTEPS,
            eval_episodes=N_EVAL_EPISODES
        )
        OPTIMIZED_HYPERPARAMS.update(best_params)
        print(f"Best Hyperparameters Found: {best_params}\n")

        # ============================
        # Step 3: Train with Optimized Params
        # ============================
        print("=== Training with Optimized Hyperparameters ===\n")
        callback_optimized = ProgressCallback()
        ppo_optimized = PPOAgent(env_id=ENV_ID, hyperparams=OPTIMIZED_HYPERPARAMS, verbose=1)
        ppo_optimized.create_model()

        # Evaluate before training with optimized params (Random Agent)
        print("Evaluating before training (Optimized Params)...")
        mean_reward, std_reward = ppo_optimized.evaluate(n_eval_episodes=N_EVAL_EPISODES)
        print(f"Stats before training (Optimized Params):\nMean Reward: {mean_reward:.2f} +/- {std_reward}\n")

        # Train and evaluate with optimized params
        train_and_evaluate(ppo_optimized, callback_optimized, TRAINING_TIMESTEPS, 'optimized')
    else:
        print("=== Hyperparameter Search Skipped ===\n")

    # ============================
    # Final Cleanup
    # ============================
    print("Closing environments...")
    ppo_default.env.close()
    if args.search:
        ppo_optimized.env.close()
    print("All environments closed.")


if __name__ == "__main__":
    main()
