import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
from torchvision.transforms import Normalize

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class Critic(nn.Module):
    def __init__(self, device, num_outputs):
        super(Critic, self).__init__()

        self.device = device

        self.architecture_network = nn.Sequential(
            nn.Conv2d(1, 50, (1,5)),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 5, (1,1)),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(5, 2, (1,1)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            Flatten()
        )

        self.weights_network = nn.Sequential(
            nn.Conv2d(6,100, (1,1000)),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100,10,(1,1)),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 5, (1, 1)),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            Flatten()
        )

        self.activation_network = nn.Sequential(
            nn.Conv2d(6,100, (1,1000)),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100,10,(1,1)),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 5, (1, 1)),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            Flatten()
        )

        self.architecture_layer = nn.Sequential(
            nn.Conv2d(1, 10, (1,5)),
            nn.ReLU(),
            nn.Conv2d(10, 10, (1,1)),
            nn.ReLU(),
            nn.Conv2d(10,10, (1,1)),
            nn.ReLU(),
            Flatten()
        )

        self.weights_layer = nn.Sequential(
            nn.Conv2d(1, 500, (6,1000)),
            nn.ReLU(),
            nn.Conv2d(500, 500, (1,1)),
            nn.ReLU(),
            nn.Conv2d(500,20,(1,1)),
            nn.ReLU(),
            Flatten()
        )

        self.activation_layer = nn.Sequential(
            nn.Conv2d(1, 500, (6, 1000)),
            nn.ReLU(),
            nn.Conv2d(500, 500, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(500, 20, (1, 1)),
            nn.ReLU(),
            Flatten()
        )


        self.critic = nn.Sequential(
            nn.Linear(170, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x):
        fm_splitted = self.split_fm(x)

        arch_net_features = self.architecture_network(self.convert_to_tensor(fm_splitted[0]).unsqueeze(0))
        weights_net_features = self.weights_network(self.convert_to_tensor(fm_splitted[2]))
        activations_net_features = self.activation_network(self.convert_to_tensor(fm_splitted[4]))

        arch_layer_features = self.architecture_layer(self.convert_to_tensor(fm_splitted[1]).unsqueeze(0).unsqueeze(0))
        weights_layer_features = self.weights_layer(self.convert_to_tensor(fm_splitted[3]).unsqueeze(0))
        activations_layer_features = self.activation_layer(self.convert_to_tensor(fm_splitted[5]).unsqueeze(0))

        features_concat = torch.cat((
            arch_net_features,
            weights_net_features,
            activations_net_features,
            arch_layer_features,
            weights_layer_features,
            activations_layer_features
        ),1)

        value = self.critic(features_concat)
        return value

    def convert_to_tensor(self, np_ar) -> torch.Tensor:
        return torch.Tensor(np_ar).to(self.device).float().unsqueeze(0)

    def split_fm(self, fm):
        arc = fm[0]
        net_arc = arc[0]
        layer_arch = arc[1]

        activations = fm[1]
        net_ac = activations[0]
        layer_ac = activations[1]

        weights = fm[2]
        net_weights = weights[0]
        layer_weights = weights[1]

        return (net_arc, layer_arch, net_ac, layer_ac, net_weights, layer_weights)

