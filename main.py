import os

from src.NetworkEnv import NetworkEnv


def load_models_path(main_path):
    model_paths = []

    for root, dirs, files in os.walk(main_path):
        if ('X_to_train.csv' not in files):
            continue
        train_data_path = root + '/X_to_train.csv'

        model_files = list(filter(lambda file_name: file_name.endswith('.pt'), files))
        model_paths.append((train_data_path, *model_files))

    return model_paths


models_path = load_models_path()

network_env = NetworkEnv(models_path)
