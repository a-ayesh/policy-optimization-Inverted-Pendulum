# Training settings
TRAINING_TIMESTEPS = 100_000
N_STEPS = 2_048
N_EVAL_EPISODES = 10
ENV_ID = 'InvertedPendulum-v5'
LOG_DIR = 'logs/ppo'

DEFAULT_HYPERPARAMS = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'n_steps': 2048
}

# Placeholder for optimized hyperparameters (to be updated after search)
OPTIMIZED_HYPERPARAMS = {}

# Hyperparameter search space for Random Search
HYPERPARAMETER_RANGES = {
    'learning_rate': [1e-5, 1e-4, 1e-3],
    'gamma': [0.9, 0.99, 0.999],
    'clip_range': [0.1, 0.2, 0.3],
    'ent_coef': [0.0, 0.01],
    'n_steps': [128, 256, 512, 1024, 2048]
}
