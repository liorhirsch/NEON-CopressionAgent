from typing import Dict


class ConfigurationValues():
    device: str
    num_epoch: int
    num_actions: int
    is_learn_new_layers_only: bool
    total_allowed_accuracy_reduction: int
    action_to_compression_rate : Dict[int, float]

    def __init__(self, device, action_to_compression_rate, num_epoch = 100, num_actions = 4, is_learn_new_layers_only = False, total_allowed_accuracy_reduction = 1) -> None:
        self.action_to_compression_rate = action_to_compression_rate
        self.device = device
        self.num_epoch = num_epoch
        self.num_actions = num_actions
        self.is_learn_new_layers_only = is_learn_new_layers_only
        self.total_allowed_accuracy_reduction = total_allowed_accuracy_reduction

