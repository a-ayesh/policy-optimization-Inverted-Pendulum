from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback

class ProgressCallback(BaseCallback):
    """
    Custom callback for storing training metrics such as returns, losses, and value losses.
    """

    def __init__(self):
        super(ProgressCallback, self).__init__()
        self.returns = []
        # A dictionary mapping metric_name -> list of values
        self.metric_history = defaultdict(list)

    def _on_step(self) -> bool:
        """
        Called at each step during training. Return True to continue, False to stop early.
        """
        # 1. Capture episode rewards if the environment ended an episode
        infos = self.locals.get("infos", [])
        if infos and "episode" in infos[0]:
            episode_reward = infos[0]["episode"]["r"]
            self.returns.append(episode_reward)

        # 2. Capture any logger metrics
        logger = getattr(self.model, "logger", None)
        if logger:
            for key, value in logger.name_to_value.items():
                if value is None:
                    continue
                # Typically keys look like 'train/actor_loss', 'train/value_loss', etc.
                if key.startswith("train/"):
                    metric_name = key.split("/", 1)[1]  # e.g. 'actor_loss'
                    self.metric_history[metric_name].append(value)

        return True
