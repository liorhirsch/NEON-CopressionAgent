import os

from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX
from src.NetworkEnv import NetworkEnv


def load_models_path(main_path):
    model_paths = []

    for root, dirs, files in os.walk(main_path):
        if ('X_to_train.csv' not in files):
            continue
        train_data_path = root + '/X_to_train.csv'

        model_files = list(map(lambda file: os.path.join(root, file),
                               filter(lambda file_name: file_name.endswith('.pt'), files)))
        model_paths.append((train_data_path, model_files))

    return model_paths


models_path = load_models_path("./Fully Connected Training/")

network_env = NetworkEnv(models_path)

network_env.reset()
network_env.step(0.5)
