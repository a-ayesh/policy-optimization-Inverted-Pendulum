from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from agents.base_agent import BaseAgent


class TD3Agent(BaseAgent):
    """
    TD3 Agent implementation.
    """

    def __init__(self, env_id: str, hyperparams: dict, verbose: int = 1):
        """
        Initializes the TD3Agent.

        Args:
            env_id (str): Gym environment ID.
            hyperparams (dict): Hyperparameters for the TD3 agent.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        super(TD3Agent, self).__init__(env_id, hyperparams, verbose)

    def create_model(self):
        """
        Creates the TD3 model with the given hyperparameters.
        """
        self.model = TD3(
            "MlpPolicy",
            self.env,
            verbose=self.verbose,
            **self.hyperparams
        )

    def train(self, total_timesteps: int, callback):
        """
        Trains the TD3 model.
        """
        if self.model is None:
            self.create_model()
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def evaluate(self, n_eval_episodes: int = 10) -> tuple:
        """
        Evaluates the TD3 model.
        """
        if self.model is None:
            raise ValueError("Model has not been created or trained yet.")
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=n_eval_episodes
        )
        return mean_reward, std_reward
