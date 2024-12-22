# Global
TRAINING_TIMESTEPS = 100_000
N_EVAL_EPISODES = 10
ENV_ID = 'InvertedPendulum-v5'
LOG_DIR = 'logs'

# =========== PPO ===========

PPO_DEFAULT_HYPERPARAMS = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'n_steps': 2048
}

PPO_OPTIMIZED_HYPERPARAMS = {}

PPO_HYPERPARAMETER_RANGES = {
    'learning_rate': [1e-5, 1e-4, 1e-3],
    'gamma': [0.9, 0.99, 0.999],
    'clip_range': [0.1, 0.2, 0.3],
    'ent_coef': [0.0, 0.01],
    'n_steps': [128, 256, 512, 1024, 2048]
}

# =========== SAC ===========

SAC_DEFAULT_HYPERPARAMS = {
    'learning_rate': 3e-4,
    'buffer_size': 1000000,
    'tau': 0.02,
    'gamma': 0.99,
    'train_freq': 64
}

SAC_OPTIMIZED_HYPERPARAMS = {}

SAC_HYPERPARAMETER_RANGES = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'tau': [0.01, 0.02, 0.05],
    'gamma': [0.95, 0.99, 0.999],
    'train_freq': [32, 64, 128],
    'buffer_size': [100000, 500000, 1000000]
}

# =========== A2C ===========

A2C_DEFAULT_HYPERPARAMS = {
    'learning_rate': 7e-4,
    'n_steps': 5,
    'gamma': 0.99,
    'gae_lambda': 1.0
}

A2C_OPTIMIZED_HYPERPARAMS = {}

A2C_HYPERPARAMETER_RANGES = {
    'learning_rate': [1e-4, 7e-4, 1e-3],
    'n_steps': [5, 16, 32],
    'gamma': [0.9, 0.99, 0.999],
    'gae_lambda': [0.9, 0.95, 1.0]
}

