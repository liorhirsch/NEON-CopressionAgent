# from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX - must be import!!!!
import glob
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import sys
import argparse
from functools import partial

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from torch import nn, optim
from NetworkFeatureExtration.src.ModelWithRows import ModelWithRows
from lookahead_pruning.method import get_method
from lookahead_pruning.network.masked_modules import MaskedLinearPreTrained
from lookahead_pruning.network.masked_pre_trained_mlp import MaskedPreTrainedMLP
from lookahead_pruning.utils import is_masked_module
from lookahead_pruning.train import train as lap_train
from src.A2C_Agent_Reinforce import A2C_Agent_Reinforce

from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf
from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX
from src.ModelHandlers.ClassificationHandler import Dataset
from src.NetworkEnv import NetworkEnv
import torch.nn.utils.prune as prune

from src.utils import load_models_path, print_flush, save_times_csv, add_weight_mask_to_all_layers, \
    set_mask_to_each_layer


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

def get_linear_layer(row):
    for l in row:
        if type(l) is nn.Linear:
            return l

def get_model_layers(model):
    new_model_with_rows = ModelWithRows(model)
    linear_layers = [(get_linear_layer(x).in_features, get_linear_layer(x).out_features) for x in
                     new_model_with_rows.all_rows]
    return str(linear_layers)


def set_weights(self, weights_to_set):
    idx = 0
    for m in self.modules():
        if is_masked_module(m):
            m.weight.data = weights_to_set[idx].cpu().to(m.weight.data.device)
            idx += 1
    assert idx == len(weights_to_set)

def set_masks(self, mask_to_set):
    idx = 0
    for m in self.modules():
        if is_masked_module(m):
            m.mask = mask_to_set[idx].cpu().to(m.mask.device)
            m.weight.data *= m.mask
            idx += 1
    assert idx == len(mask_to_set)

def get_weights(self):
    weights = []
    for m in self.modules():
        if is_masked_module(m):
            weights.append(m.weight.data.cpu().detach())
    return weights

def get_masks(self):
    masks = []
    for m in self.modules():
        if is_masked_module(m):
            masks.append(m.mask.cpu().detach())
    return masks


def look_ahead_prune_model(env: NetworkEnv, iterations):
    accuracies = []
    network = MaskedPreTrainedMLP(env.current_model)
    amount_of_linear_layers = len(list(filter(lambda x: type(x) is MaskedLinearPreTrained, network.modules()))) - 1
    prune_ratios = [.5] * (amount_of_linear_layers)
    prune_ratios.append(.25)

    # prune_ratios = [.5, .5, .5, .5, .25]

    optimizer = partial(optim.Adam, lr=0.0012)
    pruning_iteration_start = 1
    pruning_iteration_end = iterations
    pretrain_iteration = 50000
    finetune_iteration = 50000
    batch_size = 60
    pruning_type = 'oneshot'

    original_network = network  # keep the original network
    original_prune_ratio = prune_ratios  # keep the original prune ratio
    pruning_method = get_method('lap')  # lap = look-ahead-pruning

    for it in range(pruning_iteration_start, pruning_iteration_end + 1):
        # get pruning ratio for current iteration
        # list for layer-wise pruning, and constant for global pruning
        if pruning_type == 'oneshot':
            network = deepcopy(original_network).cuda()
            prune_ratios = []
            for idx in range(len(original_prune_ratio)):
                prune_ratios.append(1.0 - ((1.0 - original_prune_ratio[idx]) ** it))
        elif pruning_type == 'iterative':
            prune_ratios = []
            for idx in range(len(original_prune_ratio)):
                prune_ratios.append(original_prune_ratio[idx])
        elif pruning_type == 'global':
            network = deepcopy(original_network).cuda()
            prune_ratios = [original_prune_ratio[it - 1]]
        else:
            raise ValueError('Unknown pruning_type')

        # perpare weights and masks to prune
        weights = network.get_weights()
        masks = network.get_masks()

        masks = pruning_method(weights, masks, prune_ratios)

        network.set_masks(masks)

    new_lh = env.create_learning_handler(network.cuda())
    new_lh.unfreeze_all_layers()
    train_dataset = Dataset(env.cross_validation_obj.x_train, env.cross_validation_obj.y_train)
    val_dataset = Dataset(env.cross_validation_obj.x_val, env.cross_validation_obj.y_val)

    lap_train(train_dataset, val_dataset, network, optimizer, 10000, 32)

    new_acc = new_lh.evaluate_model()

    num_params = calc_num_parameters(network)
    pruned_params = sum(list(
        map(lambda x: (x.mask == 0).sum(), filter(lambda x: hasattr(x, 'mask'), network.modules()))))
    pruned_params = int(pruned_params)

    accuracies.append((num_params, pruned_params, new_acc))

    return accuracies[-1]


def calc_num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def evaluate_model(mode, base_path, iterations):
    models_path = load_models_path(base_path, mode)
    env = NetworkEnv(models_path, StaticConf.getInstance().conf_values.can_do_more_then_one_loop)

    results = DataFrame(columns=['model', 'new_acc', 'origin_acc', 'new_param',
                                 'origin_param', 'new_model_arch', 'origin_model_arch'])

    for i in range(len(env.all_networks)):
        state = env.reset()
        origin_lh = env.create_learning_handler(env.loaded_model.model)
        origin_acc = origin_lh.evaluate_model()

        num_params, pruned_params, new_acc = look_ahead_prune_model(env, iterations)

        model_name = env.all_networks[env.net_order[env.curr_net_index - 1]][1]

        results = results.append({'model': model_name,
                                  'new_acc': new_acc,
                                  'origin_acc': origin_acc,
                                  'new_param': num_params - pruned_params,
                                  'origin_param': num_params,
                                  'new_model_arch': get_model_layers(env.current_model),
                                  'origin_model_arch': get_model_layers(env.loaded_model.model)}, ignore_index=True)

    return results


def main(dataset_name, test_name, iterations):
    actions = {
        0: 1,
        1: 0.9,
        2: 0.8,
        3: 0.7,
        4: 0.6
    }
    base_path = f"./OneDatasetLearning/Classification/{dataset_name}/"

    init_conf_values(actions,num_epoch=5)
    # prune_percentages = [.01, .05, .1, .25, .50, .60, .70, .80, .90]
    mode = 'all'
    results = evaluate_model(mode, base_path, iterations)
    results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}.csv")

def extract_args_from_cmd():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--test_name', type=str)
    # parser.add_argument('--dataset_name', type=str)
    # parser.add_argument('--learn_new_layers_only', type=bool, const=True, default=False, nargs='?')
    # parser.add_argument('--split', type=bool, const=True, default=False, nargs='?')
    # parser.add_argument('--can_do_more_then_one_loop', type=bool, const=True, default=False, nargs='?')
    parser.add_argument('--iters', type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = extract_args_from_cmd()
    all_times = []

    print_flush(f"LAP {args.iters} iters")

    all_datasets = glob.glob("./OneDatasetLearning/Classification/*")
    for idx, curr_dataset in enumerate(all_datasets):
        dataset_name = os.path.basename(curr_dataset)
        print_flush(f"{dataset_name} {idx} / {len(all_datasets)}")
        test_name = f'Agent_{dataset_name}_LAP_{args.iters}'
        now = datetime.now()
        main(dataset_name=dataset_name, test_name=test_name, iterations=args.iters)
        total_time = (datetime.now() - now).total_seconds()
        all_times.append(total_time)

    save_times_csv(test_name, all_times, all_datasets)



