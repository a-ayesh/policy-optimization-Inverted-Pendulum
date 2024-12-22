from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from agents.base_agent import BaseAgent


class A2CAgent(BaseAgent):
    """
    A2C Agent implementation.
    """

    def __init__(self, env_id: str, hyperparams: dict, verbose: int = 1):
        """
        Initializes the A2CAgent.

        Args:
            env_id (str): Gym environment ID.
            hyperparams (dict): Hyperparameters for the A2C agent.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        super(A2CAgent, self).__init__(env_id, hyperparams, verbose)

    def create_model(self):
        """
        Creates the A2C model with the given hyperparameters.
        """
        self.model = A2C(
            "MlpPolicy",
            self.env,
            verbose=self.verbose,
            **self.hyperparams
        )

    def train(self, total_timesteps: int, callback):
        """
        Trains the A2C model.
        """
        if self.model is None:
            self.create_model()
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def evaluate(self, n_eval_episodes: int = 10) -> tuple:
        """
        Evaluates the A2C model.
        """
        if self.model is None:
            raise ValueError("Model has not been created or trained yet.")
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=n_eval_episodes
        )
        return mean_reward, std_reward
