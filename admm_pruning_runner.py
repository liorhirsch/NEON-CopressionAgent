# from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX - must be import!!!!
import argparse
from copy import deepcopy
from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX
import numpy as np
import torch
import torch.nn.utils.prune as prune
from pandas import DataFrame

from pytorch_admm_pruning.main import train as train_admm, test as test_admm, retrain as retrain_admm
from pytorch_admm_pruning.optimizer import PruneAdam
from pytorch_admm_pruning.utils import apply_l1_prune, apply_prune, print_prune
from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.ClassificationHandler import Dataset
from src.NetworkEnv import NetworkEnv
from src.utils import print_flush, load_models_path, dict2obj, get_model_layers



def init_conf_values(action_to_compression_rate=[], num_epoch=100, is_learn_new_layers_only=False,
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
        print_flush(f'{i + 1}/{len(env.all_networks)}')
        origin_lh = env.create_learning_handler(env.loaded_model.model)
        origin_acc = origin_lh.evaluate_model()

        model, checkpoint = env.loaded_model.model.cuda(), deepcopy(env.loaded_model.model.state_dict())

        admm_args = {
            'batch_size':64,
            'test_batch_size': 1000,
            'percent': [0.8, 0.92, 0.991, 0.93],
            'alpha': 5e-4,
            'rho': 1e-2,
            'l1': True,
            'l2': False,
            'num_pre_epochs': 3,
            'num_epochs': 10,
            'num_re_epochs': 3,
            'lr': 1e-3,
            'adam_epsilon': 1e-8,
            'no_cuda': False,
            'seed': 1,
            'save_model': False
        }
        kwargs = {'num_workers': 1, 'pin_memory': True}
        device = StaticConf.getInstance().conf_values.device

        admm_args = dict2obj(admm_args)

        cv = origin_lh.cross_validation_obj

        train_dataset = Dataset(cv.x_train, cv.y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=admm_args.batch_size, shuffle=True)

        val_dataset = Dataset(cv.x_val, cv.y_val)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=admm_args.batch_size, shuffle=True)

        model = env.loaded_model.model.cuda()
        optimizer = PruneAdam(model.named_parameters(), lr=admm_args.lr, eps=admm_args.adam_epsilon)
        train_admm(admm_args, model, device, train_loader, val_loader, optimizer)

        mask = apply_l1_prune(model, device, admm_args) if admm_args.l1 else apply_prune(model, device, admm_args)
        total_weights, pruned_weights = print_prune(model)
        test_admm(admm_args, model, device, val_loader)

        retrain_admm(admm_args, model, mask, device, train_loader, val_loader, optimizer)

        pruned_model_lh = env.create_learning_handler(model)
        pruned_acc = pruned_model_lh.evaluate_model()


        model_name = env.all_networks[env.net_order[env.curr_net_index - 1]][1]

        results = results.append({'model': model_name,
                                  'new_acc': pruned_acc,
                                  'origin_acc': origin_acc,
                                  'new_param': pruned_weights,
                                  'origin_param': total_weights,
                                  'new_model_arch': get_model_layers(model),
                                  'origin_model_arch': get_model_layers(env.loaded_model.model)}, ignore_index=True)

    return results


def set_mask_to_each_layer(pruned_linear_layers):
    for curr_l in pruned_linear_layers:
        curr_l.weight_mask = torch.Tensor(np.array(~(curr_l.weight == 0).cpu(), dtype=float))


def add_weight_mask_to_all_layers(pruned_linear_layers):
    for l in pruned_linear_layers:
        prune.random_unstructured(l, name="weight", amount=0.0)


def main(dataset_name, test_name):
    base_path = f"./OneDatasetLearning/Classification/{dataset_name}/"

    init_conf_values()

    mode = 'test'
    results = evaluate_model(mode, base_path)
    results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}_admm.csv")

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
    test_name = f'ADMM_{args.dataset_name}'

    main(dataset_name=args.dataset_name, test_name=test_name)
