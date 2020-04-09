import os

import torch

from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX
from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf
from src.Model.ActorCritic import ActorCritic
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


def init_conf_values():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cv = ConfigurationValues(device)
    StaticConf(cv)

init_conf_values()
models_path = load_models_path("./Fully Connected Training/")

network_env = NetworkEnv(models_path)
network_env.reset()
fm, reward, done = network_env.step(0.5)

device = StaticConf.getInstance().conf_values.device
num_actions = StaticConf.getInstance().conf_values.num_actions
ActorCritic(device, num_actions)(fm)