import os

import torch
import numpy as np
# from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX - should be import!!!!
from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX
from src.A2C_Agent import A2C_Agent
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


torch.manual_seed(0)
np.random.seed(0)


init_conf_values()
models_path = load_models_path("./Fully Connected Training/")
agent = A2C_Agent(models_path)
agent.train()
torch.save(agent.actor_critic_model.state_dict(), "ac_model.pt")