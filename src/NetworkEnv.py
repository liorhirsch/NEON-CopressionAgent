import numpy as np


class NetworkEnv:
    def __init__(self, networks_path):
        self.networks_path = networks_path

    def reset(self):
        """
        Reset environment with a random network and its train data
        :return: state includes the network and the train data
        """
        selected_net_path = np.random.choice(self.networks_path, 1)[0]



    def step(self, action):
        """
        Compress the network according to the action
        :param action: compression rate
        :return: next_state, reward
        """
