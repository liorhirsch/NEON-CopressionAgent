import os
from datetime import datetime
from os.path import join
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import prune

from NetworkFeatureExtration.src.ModelWithRows import ModelWithRows
import numpy as np

def print_flush(msg):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"{dt_string} -- {msg}", flush=True)


def load_models_path(main_path, mode='train'):
    model_paths = []

    for root, dirs, files in os.walk(main_path):
        if ('X_train.csv' not in files):
            continue
        train_data_path = join(root, 'X_train.csv')

        if mode == 'train':
            model_names = pd.read_csv(join(root, 'train_models.csv'))['0'].to_numpy()
        elif mode == 'test':
            model_names = pd.read_csv(join(root, 'test_models.csv'))['0'].to_numpy()
        else:
            model_names = files

        model_files = list(map(lambda file: os.path.join(root, file),
                               filter(lambda file_name: file_name.endswith('.pt') and file_name in model_names, files)))
        model_paths.append((train_data_path, model_files))

    return model_paths


def split_dataset_to_train_test(path):
    models_path = load_models_path(path, 'all')
    all_models = models_path[0][1]
    all_models = list(map(os.path.basename, all_models))
    train_models, test_models = train_test_split(all_models, test_size=0.2)

    df_train = pd.DataFrame(data=train_models)
    df_train.to_csv(path + "train_models.csv")

    df_test = pd.DataFrame(data=test_models)
    df_test.to_csv(path + "test_models.csv")


def get_linear_layer(row):
    for l in row:
        if type(l) is nn.Linear:
            return l


def get_model_layers_str(model):
    new_model_with_rows = ModelWithRows(model)
    linear_layers = [(get_linear_layer(x).in_features, get_linear_layer(x).out_features) for x in
                     new_model_with_rows.all_rows]
    return str(linear_layers)


def get_model_layers(model):
    new_model_with_rows = ModelWithRows(model)
    return [get_linear_layer(x) for x in new_model_with_rows.all_rows]


def dict2obj(d):
    # checking whether object d is a
    # instance of class list
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]

        # if d is not a instance of dict then
    # directly object is returned
    if not isinstance(d, dict):
        return d

        # declaring a class

    class C:
        pass

    # constructor of the class passed to obj
    obj = C()

    for k in d:
        obj.__dict__[k] = dict2obj(d[k])

    return obj

def add_weight_mask_to_all_layers(pruned_linear_layers):
    for l in pruned_linear_layers:
        prune.random_unstructured(l, name="weight", amount=0.0)

def set_mask_to_each_layer(pruned_linear_layers):
    for curr_l in pruned_linear_layers:
        curr_l.weight_mask = torch.Tensor(np.array(~(curr_l.weight == 0).cpu(), dtype=float))
