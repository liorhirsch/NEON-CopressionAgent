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
from NetworkFeatureExtration.src.ModelWithRows import ModelWithRows
from src.A2C_Agent_Reinforce import A2C_Agent_Reinforce

from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf
from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX
from src.NetworkEnv import NetworkEnv
from src.utils import print_flush, load_models_path, split_dataset_to_train_test, get_model_layers_str


def init_conf_values(action_to_compression_rate, num_epoch=100, is_learn_new_layers_only=False,
                     total_allowed_accuracy_reduction=1, can_do_more_then_one_loop=False, prune=False):
    if not torch.cuda.is_available():
        sys.exit("GPU was not allocated!!!!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_flush(f"device is {device}")
    print_flush(f"device name is {torch.cuda.get_device_name(0)}")

    num_actions = len(action_to_compression_rate)
    MAX_TIME_TO_RUN = 60 * 60 * 24 * 2.5
    cv = ConfigurationValues(device, action_to_compression_rate=action_to_compression_rate, num_actions=num_actions,
                             num_epoch=num_epoch,
                             is_learn_new_layers_only=is_learn_new_layers_only,
                             total_allowed_accuracy_reduction=total_allowed_accuracy_reduction,
                             can_do_more_then_one_loop=can_do_more_then_one_loop,
                             MAX_TIME_TO_RUN=MAX_TIME_TO_RUN,
                             prune=prune)
    StaticConf(cv)


torch.manual_seed(0)
np.random.seed(0)


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
        print_flush(i)
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
                                  'new_model_arch': get_model_layers_str(env.current_model),
                                  'origin_model_arch': get_model_layers_str(env.loaded_model.model)}, ignore_index=True)

    return results


def main(dataset_name, is_learn_new_layers_only, test_name,
         total_allowed_accuracy_reduction, is_to_split_cv=False, can_do_more_then_one_loop=False, prune=False):
    actions = {
        0: 1,
        1: 0.9,
        2: 0.8,
        3: 0.7,
        4: 0.6
    }
    base_path = f"./OneDatasetLearning/Classification/{dataset_name}/"

    init_conf_values(actions, is_learn_new_layers_only=is_learn_new_layers_only, num_epoch=100,
                     total_allowed_accuracy_reduction=total_allowed_accuracy_reduction,
                     can_do_more_then_one_loop=can_do_more_then_one_loop,
                     prune=prune)
    models_path = load_models_path(base_path, 'train')

    agent = A2C_Agent_Reinforce(models_path, test_name)
    print_flush("Starting training")
    agent.train()
    print_flush("Done training")

    print_flush("Starting evaluate train datasets")

    mode = 'test'
    results = evaluate_model(mode, base_path, agent)
    results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}.csv")

    print_flush("DONE evaluate train datasets")

    print_flush("Starting evaluate test datasets")
    mode = 'train'
    results = evaluate_model(mode, base_path, agent)
    results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}.csv")
    print_flush("DONE evaluate test datasets")


def extract_args_from_cmd():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--test_name', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--learn_new_layers_only', type=bool, const=True, default=True, nargs='?')
    parser.add_argument('--split', type=bool, const=True, default=False, nargs='?')
    parser.add_argument('--allowed_reduction_acc', type=int, nargs='?')
    parser.add_argument('--can_do_more_then_one_loop', type=bool, const=True, default=False, nargs='?')
    parser.add_argument('--prune', type=bool, const=True, default=False, nargs='?')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = extract_args_from_cmd()
    print_flush(args)
    with_loops = '_with_loop' if args.can_do_more_then_one_loop else ""
    test_name = f'Agent_{args.dataset_name}_learn_new_layers_only_{args.learn_new_layers_only}_acc_reduction_{args.allowed_reduction_acc}{with_loops}'
    main(dataset_name=args.dataset_name, is_learn_new_layers_only=args.learn_new_layers_only, test_name=test_name,
         is_to_split_cv=args.split,
         total_allowed_accuracy_reduction=args.allowed_reduction_acc,
         can_do_more_then_one_loop=args.can_do_more_then_one_loop,
         prune=args.prune)
