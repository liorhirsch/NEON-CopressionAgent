from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from src.CrossValidationObject import CrossValidationObject


class BasicHandler():
    def __init__(self, model: nn.Module, loss_function: _Loss, optimizer: Optimizer,
                 cross_validation_obj: CrossValidationObject):
        self.model = model
        self.cross_validation_obj = cross_validation_obj
        self.loss_func = loss_function
        self.optimizer = optimizer

    def evaluate_model(self) -> float:
        pass

    def train_model(self):
        pass

    def freeze_layers(self, layers_to_freeze):
        for curr_l in self.model.parameters():
            if id(curr_l) in layers_to_freeze:
                curr_l.requires_grad = True
            else:
                curr_l.requires_grad = False

