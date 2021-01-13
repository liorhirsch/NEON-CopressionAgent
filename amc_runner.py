# from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX - must be import!!!!
import glob
import os
from collections import namedtuple
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
from amc.amc_search import train
from amc.env.channel_pruning_env import ChannelPruningEnv
from amc.lib.agent import DDPG
from src.A2C_Agent_Reinforce import A2C_Agent_Reinforce

from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf
from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX
from src.NetworkEnv import NetworkEnv
import torch.nn.utils.prune as prune

from src.utils import get_model_layers_str, print_flush, load_models_path, dict2obj, get_model_layers, \
    add_weight_mask_to_all_layers, set_mask_to_each_layer


def init_conf_values(action_to_compression_rate, num_epoch=100, is_learn_new_layers_only=False,
                     total_allowed_accuracy_reduction=1, can_do_more_then_one_loop=False):
    if not torch.cuda.is_available():
        sys.exit("GPU was not allocated!!!!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_flush(f"device is {device}")
    print_flush(f"device name is {torch.cuda.get_device_name(0)}")

    num_actions = len(action_to_compression_rate)
    cv = ConfigurationValues(device, action_to_compression_rate=action_to_compression_rate, num_actions=num_actions,
                             num_epoch=num_epoch,
                             is_learn_new_layers_only=is_learn_new_layers_only,
                             total_allowed_accuracy_reduction=total_allowed_accuracy_reduction,
                             can_do_more_then_one_loop=can_do_more_then_one_loop)
    StaticConf(cv)


torch.manual_seed(0)
np.random.seed(0)


def calc_num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_quantity_of_zeros_in_layer(l):
    return float((l.weight == 0).sum())


def get_total_weights_of_layer(l):
    w = l.weight.shape
    return w[0] * w[1]


def evaluate_model(mode, base_path):
    models_path = load_models_path(base_path, mode)
    env = NetworkEnv(models_path, StaticConf.getInstance().conf_values.can_do_more_then_one_loop)

    results = DataFrame(columns=['model', 'new_acc', 'origin_acc', 'new_param',
                                 'origin_param', 'new_model_arch', 'origin_model_arch'])

    for i in range(len(env.all_networks)):
        env.reset()
        print_flush(i)
        origin_lh = env.create_learning_handler(env.loaded_model.model)
        origin_acc = origin_lh.evaluate_model()




        model, checkpoint = env.loaded_model.model.cuda(), deepcopy(env.loaded_model.model.state_dict())

        for i in range(4):
            original_lh = env.create_learning_handler(model)
            pruned_acc_before_ft = original_lh.evaluate_model()

            amc_args = {
                'model': "custom",
                'data_root': None,
                "lbound": 0.1,
                "rbound": 1.,
                "use_real_val": True,
                "acc_metric": 'acc1',
                "reward": 'acc_flops_reward',
                'n_calibration_batches': 60,
                'n_points_per_layer': 10,
                'channel_round': 8,
                'hidden1': 300,
                'hidden2': 300,
                'lr_c': 1e-3,
                'lr_a': 1e-4,
                'warmup': 100,
                'discount': 1.,
                'bsize': 64,
                'rmsize': 100,
                'window_length': 1,
                'tau': 0.01,
                'init_delta': 0.5,
                'delta_decay': 0.95,
                'max_episode_length': 1e9,
                'output': './logs',
                'debug': False,
                'init_w': 0.003,
                'train_episode': 800,
                'epsilon': 50000,
                'seed': 1,
                'n_gpu': 1,
                'n_worker': 16,
                'data_bsize': 50,
                'resume': 'default',
                'ratios': None,
                'channels': None,
                'export_path': None,
                'use_new_input': False,
                'job': 'train',
            }

            amc_args = dict2obj(amc_args)
            train_amc_env = ChannelPruningEnv(model, checkpoint, env.cross_validation_obj,
                                              preserve_ratio=0.2,
                                              batch_size=32,
                                              args=amc_args, export_model=False, use_new_input=False)

            nb_states = train_amc_env.layer_embedding.shape[1]
            nb_actions = 1  # just 1 action here

            agent = DDPG(nb_states, nb_actions, amc_args)
            train(amc_args.train_episode, agent, train_amc_env, amc_args.output, amc_args)

            train_amc_env.reset()

            for r in train_amc_env.best_strategy:
                train_amc_env.step(r)

            pruned_linear_layers = get_model_layers(train_amc_env.model)
            mask = list(map(lambda x: torch.Tensor(np.array((x.weight != 0).cpu(), dtype=float)), pruned_linear_layers))

            model.load_state_dict(checkpoint)
            pruned_linear_layers = get_model_layers(train_amc_env.model)
            add_weight_mask_to_all_layers(pruned_linear_layers)

            # set_mask_to_each_layer
            set_mask_to_each_layer(pruned_linear_layers, mask)

            # pruned_weights = sum(map(get_quantity_of_zeros_in_layer, pruned_linear_layers))
            pruned_weights = np.sum(list(map(lambda x: int(x.sum().cpu().detach().numpy()), mask)))
            total_weights = sum(map(get_total_weights_of_layer, pruned_linear_layers))

            pruned_lh = env.create_learning_handler(model)
            pruned_lh.model.cuda()
            pruned_lh.train_model()
            pruned_acc = pruned_lh.evaluate_model()

            new_model_layers = get_model_layers(model)
            for curr_layer in new_model_layers:
                curr_layer.weight = torch.where(curr_layer.weight_mask == 0, curr_layer.weight_mask, curr_layer.weight)

            checkpoint = deepcopy(model.state_dict())





        # model = train_amc_env.model.cuda()
        # checkpoint = deepcopy(train_amc_env.model.state_dict())

        model_name = env.all_networks[env.net_order[env.curr_net_index - 1]][1]

        results = results.append({'model': model_name,
                                  'new_acc': max(pruned_acc, pruned_acc_before_ft),
                                  'origin_acc': origin_acc,
                                  'new_param': pruned_weights,
                                  'origin_param': total_weights,
                                  'new_model_arch': get_model_layers(train_amc_env.model),
                                  'origin_model_arch': get_model_layers(env.loaded_model.model)}, ignore_index=True)

    return results


def main(dataset_name, test_name):
    actions = {
        0: 1,
        1: 0.9,
        2: 0.8,
        3: 0.7,
        4: 0.6
    }
    base_path = f"./OneDatasetLearning/Classification/{dataset_name}/"

    init_conf_values(actions)

    mode = 'test'
    results = evaluate_model(mode, base_path)
    results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}_amc.csv")

    # mode = 'train'
    # results = evaluate_model(mode, base_path)
    # results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}_amc.csv")


def extract_args_from_cmd():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--test_name', type=str)
    parser.add_argument('--dataset_name', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = extract_args_from_cmd()
    sys.setrecursionlimit(10000)
    test_name = f'AMC2_4iters_{args.dataset_name}'

    main(dataset_name=args.dataset_name, test_name=test_name)
