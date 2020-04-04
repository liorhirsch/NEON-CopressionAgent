import numpy as np
from torch import nn

from NetworkFeatureExtration.src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX
from NetworkFeatureExtration.src.main import get_fm_for_model_and_layer, load_model_and_data


class NetworkEnv:
    def __init__(self, networks_path):
        self.networks_path = networks_path

    def reset(self):
        """
        Reset environment with a random network and its train data
        :return: state includes the network and the train data
        """
        self.layer_index = 1
        selected_net_group_index = np.random.choice(len(self.networks_path), 1)[0]
        selected_net_group = self.networks_path[selected_net_group_index]
        selected_net_path = np.random.choice(selected_net_group[1], 1)[0]
        data_path = selected_net_group[0]

        self.model, self.data = load_model_and_data(selected_net_path, data_path)
        self.feature_extractor = FeatureExtractor(self.model, self.data._values)
        fm = self.feature_extractor.extract_features(self.layer_index)
        return fm

    def step(self, action):
        """
        Compress the network according to the action
        :param action: compression rate
        :return: next_state, reward
        """
        # Create new layer with smaller size
        # Create new model with the new layer
        new_model = self.create_new_model(action)




        return new_model


        # train network
        # calculate accuracy on the new model
        # compute reward
        # get FM for the new model and the next layer.

    def get_linear_layer(self, row):
        for l in row:
            if type(l) is nn.Linear:
                return l

    def create_new_model(self, action):
        model_with_rows = self.feature_extractor.model_with_rows
        prev_layer_to_change = self.get_linear_layer(model_with_rows.all_rows[self.layer_index - 1])
        layer_to_change = self.get_linear_layer(model_with_rows.all_rows[self.layer_index])
        new_model_layers = []
        new_size = int(np.ceil(action * layer_to_change.in_features))
        for l in model_with_rows.all_layers:
            if (l is prev_layer_to_change):
                new_model_layers.append(nn.Linear(l.in_features, new_size))
            elif l is layer_to_change:
                new_model_layers.append(nn.Linear(new_size, l.out_features))
            else:
                new_model_layers.append(l)

        new_model = nn.Sequential(*new_model_layers)
        return new_model

