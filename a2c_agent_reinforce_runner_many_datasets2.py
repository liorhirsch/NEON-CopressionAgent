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


def init_conf_values(action_to_compression_rate, num_epoch=100, is_learn_new_layers_only=False,
                     total_allowed_accuracy_reduction=1, can_do_more_then_one_loop=False, prune=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_actions = len(action_to_compression_rate)
    cv = ConfigurationValues(device, action_to_compression_rate=action_to_compression_rate, num_actions=num_actions,
                             num_epoch=num_epoch,
                             is_learn_new_layers_only=is_learn_new_layers_only,
                             total_allowed_accuracy_reduction=total_allowed_accuracy_reduction,
                             can_do_more_then_one_loop=can_do_more_then_one_loop,
                             prune = prune)
    StaticConf(cv)


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


def get_linear_layer(row):
    for l in row:
        if type(l) is nn.Linear:
            return l


def get_model_layers(model):
    new_model_with_rows = ModelWithRows(model)
    linear_layers = [(get_linear_layer(x).in_features, get_linear_layer(x).out_features) for x in
                     new_model_with_rows.all_rows]
    return str(linear_layers)


def evaluate_model(mode, base_path, agent):
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
        done = False

        while not done:
            # dist, value = agent.actor_critic_model(state)
            value = agent.critic_model(state)
            dist = agent.actor_model(state)

            action = dist.sample()
            compression_rate = action_to_compression[action.cpu().numpy()[0]]
            next_state, reward, done = env.step(compression_rate)
            state = next_state

        new_lh = env.create_learning_handler(env.current_model)
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


def main(is_learn_new_layers_only, test_name,
         total_allowed_accuracy_reduction, is_to_split_cv=False, can_do_more_then_one_loop = False,
         prune = False, dataset_split_seed = 0):
    base_path = f"./OneDatasetLearning/Classification/"
    datasets = list(map(os.path.basename, glob.glob(join(base_path, "*"))))
    train_datasets, test_datasets = train_test_split(datasets, test_size = 0.2, random_state=dataset_split_seed)
    print("train datasets = ", train_datasets)
    print("test datasets = ", test_datasets)

    actions = {
        0: 1,
        1: 0.9,
        2: 0.8,
        3: 0.7,
        4: 0.6
    }

    if prune:
        num_epoch = 10
    else:
        num_epoch = 100

    init_conf_values(actions, is_learn_new_layers_only=is_learn_new_layers_only, num_epoch=num_epoch,
                     total_allowed_accuracy_reduction=total_allowed_accuracy_reduction,
                     can_do_more_then_one_loop=can_do_more_then_one_loop, prune=prune)

    train_models_path = [load_models_path(join(base_path, dataset_name), 'train') for dataset_name in train_datasets]
    test_models_path = [load_models_path(join(base_path, dataset_name), 'train') for dataset_name in test_datasets]
    flatten = lambda l: [item for sublist in l for item in sublist]

    train_models_path = flatten(train_models_path)
    test_models_path = flatten(test_models_path)

    agent = A2C_Agent_Reinforce(train_models_path)
    print("Starting training")
    agent.train()
    print("Done training")

    print("Starting evaluate train datasets")

    for d in train_datasets:
        mode = 'test'
        results = evaluate_model(mode, join(base_path, d), agent)
        results.to_csv(f"./models/Reinforce_One_Dataset/results_{d}_{test_name}_{mode}_trained_dataset.csv")

        mode = 'train'
        results = evaluate_model(mode, join(base_path, d), agent)
        results.to_csv(f"./models/Reinforce_One_Dataset/results_{d}_{test_name}_{mode}_trained_dataset.csv")

    print("Starting evaluate test datasets")
    for d in test_datasets:
        mode = 'test'
        results = evaluate_model(mode, join(base_path, d), agent)
        results.to_csv(f"./models/Reinforce_One_Dataset/results_{d}_{test_name}_{mode}_unseen_dataset.csv")

        mode = 'train'
        results = evaluate_model(mode, join(base_path, d), agent)
        results.to_csv(f"./models/Reinforce_One_Dataset/results_{d}_{test_name}_{mode}_unseen_dataset.csv")



def extract_args_from_cmd():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--test_name', type=str)
    # parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--learn_new_layers_only', type=bool, const=True, default=False, nargs='?')
    parser.add_argument('--split', type=bool, const=True, default=False, nargs='?')
    parser.add_argument('--allowed_reduction_acc', type=int, nargs='?')
    parser.add_argument('--can_do_more_then_one_loop', type=bool, const=True, default=False, nargs='?')
    parser.add_argument('--prune', type=bool, const=True, default=False, nargs='?')

    args = parser.parse_args()
    return args

print("hello")
#
#
# if __name__ == "__main__":
#     print("Starting scripttt")
#     args = extract_args_from_cmd()
#     print(args)
#     with_loops = '_with_loop' if args.can_do_more_then_one_loop else ""
#     pruned = '_pruned' if args.prune else ""
#     test_name = f'All_Datasets_Agent_learn_new_layers_only_{args.learn_new_layers_only}_acc_reduction_{args.allowed_reduction_acc}{with_loops}{pruned}'
#     print(test_name)
#     main(is_learn_new_layers_only=args.learn_new_layers_only, test_name=test_name,
#          is_to_split_cv=args.split,
#          total_allowed_accuracy_reduction=args.allowed_reduction_acc,
#          can_do_more_then_one_loop=args.can_do_more_then_one_loop,
#          prune=args.prune)
