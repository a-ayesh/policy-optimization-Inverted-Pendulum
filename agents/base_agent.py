# agents/base_agent.py

from abc import ABC, abstractmethod
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


class BaseAgent(ABC):
    """
    Abstract base class for RL agents.
    """

    def __init__(self, env_id: str, hyperparams: dict, verbose: int = 1):
        """
        Initializes the BaseAgent.

        Args:
            env_id (str): Gym environment ID.
            hyperparams (dict): Hyperparameters for the agent.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        self.env_id = env_id
        self.env = make_vec_env(env_id, n_envs=4)
        self.hyperparams = hyperparams
        self.verbose = verbose
        self.model = None

    @abstractmethod
    def create_model(self):
        """
        Creates the RL model.
        """
        pass

    @abstractmethod
    def train(self, total_timesteps: int, callback: BaseCallback):
        """
        Trains the RL model.

        Args:
            total_timesteps (int): Total training timesteps.
            callback (BaseCallback): Callback for monitoring training.
        """
        pass

    @abstractmethod
    def evaluate(self, n_eval_episodes: int = 10) -> tuple:
        """
        Evaluates the RL model.

        Args:
            n_eval_episodes (int, optional): Number of evaluation episodes. Defaults to 10.

        Returns:
            tuple: Mean reward and standard deviation.
        """
        pass
