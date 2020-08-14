# from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX - must be import!!!!

import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from scipy.stats import rankdata
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
        if ('X_to_train.csv' not in files):
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


def init_conf_values(action_to_compression_rate, num_epoch=100, is_learn_new_layers_only=False,
                     total_allowed_accuracy_reduction=1, can_do_more_then_one_loop=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_actions = len(action_to_compression_rate)
    cv = ConfigurationValues(device, action_to_compression_rate=action_to_compression_rate, num_actions=num_actions,
                             num_epoch=num_epoch,
                             is_learn_new_layers_only=is_learn_new_layers_only,
                             total_allowed_accuracy_reduction=total_allowed_accuracy_reduction,
                             can_do_more_then_one_loop=can_do_more_then_one_loop)
    StaticConf(cv)


torch.manual_seed(0)
np.random.seed(0)


def split_dataset_to_train_test(path):
    models_path = load_models_path(path, 'all')
    all_models = models_path[0][1]
    all_models = list(map(os.path.basename, all_models))
    train_models, test_models = train_test_split(all_models, test_size=0.2)

    df_train = DataFrame(data=train_models)
    df_train.to_csv(path + "train_models.csv")

    df_test = DataFrame(data=test_models)
    df_test.to_csv(path + "test_models.csv")


def get_linear_layer(row):
    for l in row:
        if type(l) is nn.Linear:
            return l


def get_model_layers(model):
    new_model_with_rows = ModelWithRows(model)
    linear_layers = [(get_linear_layer(x).in_features, get_linear_layer(x).out_features) for x in
                     new_model_with_rows.all_rows]
    return str(linear_layers)

def prune(env: NetworkEnv):
    prune_percentage = [.0, .25, .50, .60, .70, .80, .90, .95, .97, .99]
    accuracies_wp = []
    accuracies_np = []

    # Get the accuracy without any pruning
    initial_accuracy = env.original_acc
    accuracies_wp.append(initial_accuracy)
    accuracies_np.append(initial_accuracy)

    for k in prune_percentage[1:]:
        model = deepcopy(env.current_model)

        for _ in range(4):
            pruned_weights = []
            weights = model.state_dict()
            layers = list(model.state_dict())
            ranks = {}

            for l in layers[:-1]:
                data = weights[l]
                if 'Weight' in l.title() and data.ndim > 1:
                    w = np.array(data)
                    # taking norm for each neuron
                    norm = np.linalg.norm(w, axis=0)
                    # repeat the norm values to get the shape similar to that of layer weights
                    norm = np.tile(norm, (w.shape[0], 1))
                    # norm = np.tile(norm, (w.shape[0]))
                    ranks[l] = (rankdata(norm, method='dense') - 1).astype(int).reshape(norm.shape)
                    lower_bound_rank = np.ceil(np.max(ranks[l]) * k).astype(int)
                    ranks[l][ranks[l] <= lower_bound_rank] = 0
                    ranks[l][ranks[l] > lower_bound_rank] = 1
                    w = w * ranks[l]
                    data[...] = torch.from_numpy(w)
                pruned_weights.append(data)
            pruned_weights.append(weights[layers[-1]])
            new_state_dict = OrderedDict()
            for l, pw in zip(layers, pruned_weights):
                new_state_dict[l] = pw
            model.load_state_dict(new_state_dict)

            new_lh = env.create_learning_handler(model)
            new_lh.unfreeze_all_layers()
            new_lh.train_model()
            new_acc = new_lh.evaluate_model()

            accuracies_np.append(new_acc)

    return accuracies_np



def evaluate_model(mode, base_path):
    models_path = load_models_path(base_path, mode)
    env = NetworkEnv(models_path, StaticConf.getInstance().conf_values.can_do_more_then_one_loop)
    action_to_compression = {
        0: 1,
        1: 0.9,
        2: 0.8,
        3: 0.7,
        4: 0.6
    }

    results = DataFrame(columns=['model', 'new_acc', 'origin_acc', 'new_param',
                                 'origin_param', 'new_model_arch', 'origin_model_arch'])

    for i in range(len(env.all_networks)):
        print(i)
        state = env.reset()

        new_lh = env.create_learning_handler(env.current_model)
        prune(env)
        origin_lh = env.create_learning_handler(env.loaded_model.model)

        new_acc = new_lh.evaluate_model()
        origin_acc = origin_lh.evaluate_model()

        new_params = env.calc_num_parameters(env.current_model)
        origin_params = env.calc_num_parameters(env.loaded_model.model)

        model_name = env.all_networks[env.net_order[env.curr_net_index - 1]][1]

        new_model_with_rows = ModelWithRows(env.current_model)

        results = results.append({'model': model_name,
                                  'new_acc': new_acc,
                                  'origin_acc': origin_acc,
                                  'new_param': new_params,
                                  'origin_param': origin_params,
                                  'new_model_arch': get_model_layers(env.current_model),
                                  'origin_model_arch': get_model_layers(env.loaded_model.model)}, ignore_index=True)

    return results


def main(dataset_name, is_learn_new_layers_only, test_name,
         is_to_split_cv=False, can_do_more_then_one_loop = False):
    actions = {
        0: 1,
        1: 0.9,
        2: 0.8,
        3: 0.7,
        4: 0.6
    }
    base_path = f"./OneDatasetLearning/Classification/{dataset_name}/"

    if is_to_split_cv:
        split_dataset_to_train_test(base_path)

    init_conf_values(actions, is_learn_new_layers_only=is_learn_new_layers_only, num_epoch=10,
                     can_do_more_then_one_loop=can_do_more_then_one_loop)
    mode = 'test'
    results = evaluate_model(mode, base_path)
    results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}.csv")

    mode = 'train'
    results = evaluate_model(mode, base_path)
    results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}.csv")







def extract_args_from_cmd():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--test_name', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--learn_new_layers_only', type=bool, const=True, default=False, nargs='?')
    parser.add_argument('--split', type=bool, const=True, default=False, nargs='?')
    parser.add_argument('--can_do_more_then_one_loop', type=bool, const=True, default=False, nargs='?')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = extract_args_from_cmd()
    with_loops = '_with_loop' if args.can_do_more_then_one_loop else ""
    test_name = f'Agent_{args.dataset_name}_learn_new_layers_only_{args.learn_new_layers_only}_{with_loops}_Random_Actions'
    main(dataset_name=args.dataset_name, is_learn_new_layers_only=args.learn_new_layers_only,test_name=test_name,
         is_to_split_cv=args.split,
         can_do_more_then_one_loop=args.can_do_more_then_one_loop)
