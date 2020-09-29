# from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX - must be import!!!!
import glob
import os
from datetime import datetime
import sys
import argparse
from os.path import join

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch import nn
from NetworkFeatureExtration.src.ModelWithRows import ModelWithRows
from src.A2C_Agent_Reinforce import A2C_Agent_Reinforce

from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf
from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX
from src.NetworkEnv import NetworkEnv


def load_models_path(main_path, mode='train'):
    model_paths = []

    for root, dirs, files in os.walk(main_path):
        if ('X_train.csv' not in files):
            continue
        train_data_path = root + '/X_to_train.csv'

        if mode == 'train':
            model_names = pd.read_csv(root + '/train_models.csv')['0'].to_numpy()
        elif mode == 'test':
            model_names = pd.read_csv(root + '/test_models.csv')['0'].to_numpy()
        else:
            model_names = files

        model_files = list(map(lambda file: os.path.join(root, file),
                               filter(lambda file_name: file_name.endswith('.pt') and file_name in model_names, files)))
        model_paths.append((train_data_path, model_files))

    return model_paths

torch.manual_seed(0)
np.random.seed(0)


def split_dataset_to_train_test(path):
    models_path = load_models_path(path, 'all')
    all_models = models_path[0][1]
    all_models = list(map(os.path.basename, all_models))
    train_models, test_models = train_test_split(all_models, test_size=0.2)

    df_train = DataFrame(data=train_models)
    df_train.to_csv(join(path, "train_models.csv"))

    df_test = DataFrame(data=test_models)
    df_test.to_csv(join(path, "test_models.csv"))

def main():
    base_path = r"./OneDatasetLearning/Classification/"
    datasets = list(map(os.path.basename, glob.glob(join(base_path, "*"))))

    for curr_d in datasets:
        split_dataset_to_train_test(join(base_path, curr_d))


if __name__ == "__main__":
    main()
