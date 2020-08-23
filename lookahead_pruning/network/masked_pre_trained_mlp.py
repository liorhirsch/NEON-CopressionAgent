import torch.nn as nn
import torch.nn.functional as F

from ..network.base_model import BaseModel
from ..network.masked_modules import MaskedLinear, MaskedLinearPreTrained


class MaskedPreTrainedMLP(BaseModel):
	def __init__(self, network):
		super(MaskedPreTrainedMLP, self).__init__()
		layers = []
		only_linear_layers = list(filter(lambda x: type(x) is nn.Linear, network.modules()))
		depth = len(only_linear_layers)
		linear_layers = []

		for layer in range(depth):
			current_linear_layer = only_linear_layers[layer]
			linear_layers += [MaskedLinearPreTrained(current_linear_layer.in_features, current_linear_layer.out_features,
											  current_linear_layer.weight, current_linear_layer.bias)]
		linear_layers_index = 0

		for curr_layer in list(network.modules())[2:]:
			if type(curr_layer) is nn.Linear:
				layers.append(linear_layers[linear_layers_index])
				linear_layers_index += 1
			else:
				layers.append(curr_layer)


			# if layer < depth - 1:
				# layers += [nn.ReLU(inplace=True)]
		self.fc = nn.Sequential(*layers)
		self.layers = layers

	def forward(self, x):
		x = x.view(len(x), -1)
		x = self.fc(x)
		return x

	def forward_activation(self, x):
		activation = []

		x = x.view(len(x), -1)
		for layer in self.layers:
			x = layer(x)

			if isinstance(layer, nn.ReLU):
				activation.append((x.data > 0).sum(dim=0))

		return activation
