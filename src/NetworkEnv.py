import numpy as np
from torch import nn

from NetworkFeatureExtration.src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
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
        selected_net_group = np.random.choice(self.networks_path, 1)[0]
        selected_net_path = np.random.choice(selected_net_group[0], 1)[0]
        data_path = selected_net_group[1]

        self.model, self.data = load_model_and_data(selected_net_path, data_path)
        self.feature_extractor = FeatureExtractor(self.model, self.data._values)
        self.feature_extractor.extract_features(self.layer_index)

        fm = get_fm_for_model_and_layer(self.model, self.data, self.layer_index)
        return fm

    def step(self, action):
        """
        Compress the network according to the action
        :param action: compression rate
        :return: next_state, reward
        """
        # TODO - Create new layer with smaller size

        new_model = self.create_new_model(action)

        return new_model

        # create new model with the new layer
        # train network
        # calculate accuracy on the new model
        # compute reward
        # get FM for the new model and the next layer.

    def create_new_model(self, action):
        prev_layer_to_change = self.feature_extractor.all_rows[self.layer_index - 1][0]
        layer_to_change = self.feature_extractor.all_rows[self.layer_index][0]
        new_model_layers = []
        new_size = np.ceil(action * layer_to_change.in_features)
        for l in self.feature_extractor.all_layers:
            nn.Linear().out_features

            if (l is prev_layer_to_change):
                new_model_layers.append(nn.Linear(l.in_features, new_size))
            elif l is layer_to_change:
                new_model_layers.append(nn.Linear(new_size, l.out_features))
            else:
                new_model_layers.append(l)

            new_model = nn.Sequential(*new_model_layers)
        return new_model

