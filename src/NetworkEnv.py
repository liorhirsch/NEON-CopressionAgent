from typing import Dict, Type

import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn

from NetworkFeatureExtration.src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
from NetworkFeatureExtration.src.ModelClasses.LoadedModel import LoadedModel, MissionTypes
from NetworkFeatureExtration.src.ModelWithRows import ModelWithRows
from NetworkFeatureExtration.src.main import load_model_and_data
from src.Configuration.StaticConf import StaticConf
from src.CrossValidationObject import CrossValidationObject
from src.ModelHandlers.BasicHandler import BasicHandler
from src.ModelHandlers.ClassificationHandler import ClassificationHandler
from src.ModelHandlers.RegressionHandler import RegressionHandler

flatten_list = lambda l: [item for sublist in l for item in sublist]


class NetworkEnv:
    loaded_model: LoadedModel
    current_model: nn.Module

    def __init__(self, networks_path):
        self.networks_path = networks_path
        self.handler_by_mission_type: Dict[MissionTypes, Type[BasicHandler]] = {
            MissionTypes.Regression: RegressionHandler,
            MissionTypes.Classification: ClassificationHandler
        }

    def create_train_test_splits(self):
        self.cross_validation_obj = CrossValidationObject(
            *train_test_split(self.X_data.values, self.Y_data.values, test_size=0.2))

    def reset(self):
        """
        Reset environment with a random network and its train data
        :return: state includes the network and the train data
        """
        self.layer_index = 1
        selected_net_group_index = np.random.choice(len(self.networks_path), 1)[0]
        selected_net_group = self.networks_path[selected_net_group_index]
        x_path = selected_net_group[0]
        print("Selected net: ", x_path)
        selected_net_path = np.random.choice(selected_net_group[1], 1)[0]
        print("Selected net: ", selected_net_path)

        # selected_net_path = './Fully Connected Training/Classification\\fri_c0_250_5\\netX8model.pt'
        # x_path = './Fully Connected Training/Classification\\fri_c0_250_5\\X_to_train.csv'


        y_path = str.replace(x_path, 'X_to_train', 'Y_to_train')

        device = StaticConf.getInstance().conf_values.device
        self.loaded_model, self.X_data, self.Y_data = load_model_and_data(selected_net_path, x_path, y_path, device)
        self.current_model = self.loaded_model.model
        self.feature_extractor = FeatureExtractor(self.loaded_model.model, self.X_data._values)
        fm = self.feature_extractor.extract_features(self.layer_index)
        self.create_train_test_splits()
        return fm

    def step(self, action):
        """
        Compress the network according to the action
        :param action: compression rate
        :return: next_state, reward
        """
        # Create new layer with smaller size & Create new model with the new layer
        new_model = self.create_new_model(action)
        new_model_with_rows = ModelWithRows(new_model)

        learning_handler_prev_model = self.create_learning_handler(self.current_model)

        learning_handler_new_model = self.create_learning_handler(new_model)

        prev_acc = learning_handler_prev_model.evaluate_model()

        parameters_to_freeze_ids = self.build_parameters_to_freeze(new_model_with_rows)

        # train network
        learning_handler_new_model.freeze_layers(parameters_to_freeze_ids)
        learning_handler_new_model.train_model()
        new_acc = learning_handler_new_model.evaluate_model()

        # compute reward
        reward = self.compute_reward(self.current_model, learning_handler_new_model.model, new_acc, prev_acc,
                                     self.loaded_model.mission_type)

        self.layer_index += 1
        learning_handler_new_model.unfreeze_all_layers()
        self.current_model = learning_handler_new_model.model

        # get FM for the new model and the next layer.
        self.feature_extractor = FeatureExtractor(self.current_model, self.X_data._values)
        fm = self.feature_extractor.extract_features(self.layer_index)

        # Compute done
        done = self.layer_index + 1 == len(self.feature_extractor.model_with_rows.all_rows)
        return fm, reward, done

    def compute_reward(self, curr_model, new_model, new_acc, prev_acc, mission_type):
        current_model_parameter_num = self.calc_num_parameters(curr_model)
        new_model_parameter_num = self.calc_num_parameters(new_model)
        C = 1 - (new_model_parameter_num / current_model_parameter_num)
        reward = C * (2 - C)

        if (mission_type is MissionTypes.Classification):
            reward *= new_acc / prev_acc
        else:
            reward *= prev_acc / new_acc
        return reward

    def calc_num_parameters(self, model):
        return sum(p.numel() for p in model.parameters())

    def build_parameters_to_freeze(self, new_model_with_rows):
        layers_to_freeze = np.concatenate(new_model_with_rows.all_rows[self.layer_index - 1:self.layer_index + 1])
        parameters_to_freeze = flatten_list(list(map(lambda l: list(l.parameters()), layers_to_freeze)))
        parameters_to_freeze_ids = list(map(lambda l: id(l), parameters_to_freeze))
        return parameters_to_freeze_ids

    def create_learning_handler(self, new_model) -> BasicHandler:
        return self.handler_by_mission_type[self.loaded_model.mission_type](new_model,
                                                                            self.loaded_model.loss,
                                                                            self.loaded_model.optimizer,
                                                                            self.cross_validation_obj)

    def get_linear_layer(self, row):
        for l in row:
            if type(l) is nn.Linear:
                return l

    def get_batchnorm_layer(self, row):
        for l in row:
            if type(l) is nn.BatchNorm1d:
                return l

    def create_new_model(self, action):
        model_with_rows = ModelWithRows(self.current_model)

        prev_layer_to_change = self.get_linear_layer(model_with_rows.all_rows[self.layer_index - 1])
        bn_to_change = None if self.layer_index is 1 else \
            self.get_batchnorm_layer(model_with_rows.all_rows[self.layer_index - 1])

        layer_to_change = self.get_linear_layer(model_with_rows.all_rows[self.layer_index])

        new_model_layers = []
        new_size = int(np.ceil(action * layer_to_change.in_features))
        last_linear_layer = None

        for l in model_with_rows.all_layers:
            if (l is prev_layer_to_change):
                new_model_layers.append(nn.Linear(l.in_features, new_size))
            elif l is layer_to_change:
                new_model_layers.append(nn.Linear(new_size, l.out_features))
            elif self.is_to_change_bn_layer(l, last_linear_layer):
                new_model_layers.append(nn.BatchNorm1d(last_linear_layer.out_features))
            else:
                new_model_layers.append(l)

            if type(l) is nn.Linear:
                last_linear_layer = new_model_layers[-1]

        new_model = nn.Sequential(*new_model_layers)
        return new_model

    def is_to_change_bn_layer(self, curr_layer, last_linear_layer):
        return type(curr_layer) is nn.BatchNorm1d and \
               last_linear_layer is not None and \
               curr_layer.num_features is not last_linear_layer.out_features
