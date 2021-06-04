# from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX - must be import!!!!

import os
from datetime import datetime
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import prune

from NetworkFeatureExtration.src.ModelWithRows import ModelWithRows
from a2c_agent_pruning_runner import calc_num_parameters
from src.A2C_Agent_Reinforce import A2C_Agent_Reinforce
import glob

from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf
from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX
from src.NetworkEnv import NetworkEnv
from src.utils import print_flush, load_models_path


# def load_models_path(main_path, mode='train'):
#     model_paths = []
#
#     for root, dirs, files in os.walk(main_path):
#         if ('X_to_train.csv' not in files):
#             continue
#         train_data_path = root + '/X_to_train.csv'
#
#         if mode == 'train':
#             model_names = pd.read_csv(root + '/train_models.csv')['0'].to_numpy()
#         elif mode == 'test':
#             model_names = pd.read_csv(root + '/test_models.csv')['0'].to_numpy()
#         else:
#             model_names = files
#
#         model_files = list(map(lambda file: os.path.join(root, file),
#                                filter(lambda file_name: file_name.endswith('.pt') and file_name in model_names, files)))
#         model_paths.append((train_data_path, model_files))
#
#     return model_paths


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


def get_linear_layer(row):
    for l in row:
        if type(l) is nn.Linear:
            return l


def get_model_layers(model):
    new_model_with_rows = ModelWithRows(model)
    linear_layers = [(get_linear_layer(x).in_features, get_linear_layer(x).out_features) for x in
                     new_model_with_rows.all_rows]
    return str(linear_layers)


def evaluate_model(mode, base_path):
    models_path = load_models_path(base_path, mode)
    env = NetworkEnv(models_path, StaticConf.getInstance().conf_values.can_do_more_then_one_loop)
    action_to_compression = {
        # 0: 1,
        1: 0.9,
        2: 0.8,
        3: 0.7,
        4: 0.6,
    }

    results = DataFrame(columns=['model', 'new_acc', 'origin_acc', 'new_param',
                                 'origin_param', 'new_model_arch', 'origin_model_arch'])

    for i in range(len(env.all_networks)):
        print(i)
        state = env.reset()
        model = env.loaded_model.model

        origin_lh = env.create_learning_handler(model)
        origin_acc = origin_lh.evaluate_model()

        for _ in range(4):
            for l in list(model.modules()):
                if type(l) is torch.nn.Linear:
                    action = np.random.choice(list(action_to_compression.keys()), 1)[0]
                    compression_rate = action_to_compression[action]
                    prune.l1_unstructured(l, name='weight', amount=1 - compression_rate)

        origin_params = calc_num_parameters(model)
        pruned_params = sum(list(
            map(lambda x: (x.weight_mask == 1).sum(), filter(lambda x: hasattr(x, 'weight_mask'), model.modules()))))
        new_params = int(pruned_params)

        new_lh = env.create_learning_handler(env.current_model)
        new_lh.unfreeze_all_layers()
        new_lh.train_model()
        new_acc = new_lh.evaluate_model()

        model_name = env.all_networks[env.net_order[env.curr_net_index - 1]][1]

        results = results.append({'model': model_name,
                                  'new_acc': new_acc,
                                  'origin_acc': origin_acc,
                                  'new_param': new_params,
                                  'origin_param': origin_params,
                                  'new_model_arch': get_model_layers(env.current_model),
                                  'origin_model_arch': get_model_layers(env.loaded_model.model)}, ignore_index=True)

    return results


def main(dataset_name, is_learn_new_layers_only, test_name,
         can_do_more_then_one_loop=False):
    actions = {
        # 0: 1,
        1: 0.9,
        2: 0.8,
        3: 0.7,
        4: 0.6,
    }

    base_path = f"./OneDatasetLearning/Classification/{dataset_name}/"
    init_conf_values(actions, is_learn_new_layers_only=is_learn_new_layers_only, num_epoch=10,
                     can_do_more_then_one_loop=can_do_more_then_one_loop)

    mode = 'all'
    results = evaluate_model(mode, base_path)
    results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}.csv")

    # mode = 'train'
    # results = evaluate_model(mode, base_path)
    # results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}.csv")


def extract_args_from_cmd():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--test_name', type=str)
    parser.add_argument('--learn_new_layers_only', type=bool, const=True, default=True, nargs='?')
    # parser.add_argument('--split', type=bool, const=True, default=False, nargs='?')
    parser.add_argument('--can_do_more_then_one_loop', type=bool, const=True, default=True, nargs='?')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = extract_args_from_cmd()

    all_datasets = glob.glob("./OneDatasetLearning/Classification/*")

    for curr_dataset in all_datasets:
        args.dataset_name = os.path.basename(curr_dataset)
        print_flush(args.dataset_name)
        with_loops = '_with_loop' if args.can_do_more_then_one_loop else ""
        test_name = f'Agent_{args.dataset_name}_Random_pruning'
        main(dataset_name=args.dataset_name, is_learn_new_layers_only=args.learn_new_layers_only, test_name=test_name,
             can_do_more_then_one_loop=args.can_do_more_then_one_loop)
