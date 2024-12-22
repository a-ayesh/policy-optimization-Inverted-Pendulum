from stable_baselines3.common.callbacks import BaseCallback


class ProgressCallback(BaseCallback):
    """
    Custom callback for storing training metrics such as returns, losses, and value losses.
    """

    def __init__(self):
        """
        Initializes the ProgressCallback.
        """
        super(ProgressCallback, self).__init__()
        self.returns = []
        self.losses = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        self.approx_kl = []
        self.explained_variance = []
        self.std = []

    def _on_step(self) -> bool:
        """
        This method is called at each step during training.

        Returns:
            bool: Continue training.
        """
        infos = self.locals.get("infos", [])
        if infos and "episode" in infos[0]:
            episode_reward = infos[0]["episode"]["r"]
            self.returns.append(episode_reward)

        logger = getattr(self.model, "logger", None)
        if logger:
            # Retrieve and store all desired metrics if available
            for key in [
                'train/loss',
                'train/value_loss',
                'train/policy_gradient_loss',
                'train/entropy_loss',
                'train/approx_kl',
                'train/clip_fraction',
                'train/clip_range',
                'train/explained_variance',
                'train/learning_rate',
                'train/n_updates',
                'train/std'
            ]:
                value = logger.name_to_value.get(key)
                if value is not None:
                    if key == 'train/loss':
                        self.losses.append(value)
                    elif key == 'train/value_loss':
                        self.value_losses.append(value)
                    elif key == 'train/policy_gradient_loss':
                        self.policy_losses.append(value)
                    elif key == 'train/entropy_loss':
                        self.entropy_losses.append(value)
                    elif key == 'train/approx_kl':
                        self.approx_kl.append(value)
                    elif key == 'train/explained_variance':
                        self.explained_variance.append(value)
                    elif key == 'train/std':
                        self.std.append(value)

        return True
