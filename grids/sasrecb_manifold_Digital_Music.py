import optuna
from grids.core import sasrecb_manifold, generate_base_config


def generate_config(trial: optuna.Trial) -> dict:
    trial_params = sasrecb_manifold.trial_params
    fixed_params =   { "batch_size": 128,
        "learning_rate": 0.0005,
        "hidden_units": 64,
        "num_blocks": 3,
        "dropout_rate": 0.4,
        "num_heads": 1,
        "l2_emb": 0.0,
        "maxlen": 50,
        "batch_quota": None,
        "seed": 0,
        "sampler_seed": 789,
        "device": None,
        "max_epochs": 400
    }
    suggest = trial.suggest_categorical
    config = generate_base_config(trial_params, fixed_params, suggest)
    return config
