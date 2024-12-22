# agents/ppo_agent.py

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from agents.base_agent import BaseAgent
from stable_baselines3.common.env_util import make_vec_env


class PPOAgent(BaseAgent):
    """
    PPO Agent implementation.
    """

    def __init__(self, env_id: str, hyperparams: dict, verbose: int = 1):
        """
        Initializes the PPOAgent.

        Args:
            env_id (str): Gym environment ID.
            hyperparams (dict): Hyperparameters for the PPO agent.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        super(PPOAgent, self).__init__(env_id, hyperparams, verbose)

    def create_model(self):
        """
        Creates the PPO model with the given hyperparameters.
        """
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=self.verbose,
            **self.hyperparams
        )

    def train(self, total_timesteps: int, callback):
        """
        Trains the PPO model.

        Args:
            total_timesteps (int): Total training timesteps.
            callback (BaseCallback): Callback for monitoring training.
        """
        if self.model is None:
            self.create_model()
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def evaluate(self, n_eval_episodes: int = 10) -> tuple:
        """
        Evaluates the PPO model.

        Args:
            n_eval_episodes (int, optional): Number of evaluation episodes. Defaults to 10.

        Returns:
            tuple: Mean reward and standard deviation.
        """
        if self.model is None:
            raise ValueError("Model has not been created or trained yet.")
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=n_eval_episodes
        )
        return mean_reward, std_reward
