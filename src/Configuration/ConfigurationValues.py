class ConfigurationValues():
    device: str
    num_epoch: int
    num_actions: int
    is_learn_new_layers_only: bool

    def __init__(self, device, num_epoch = 100, num_actions = 4, is_learn_new_layers_only = False) -> None:
        self.device = device
        self.num_epoch = num_epoch
        self.num_actions = num_actions
        self.is_learn_new_layers_only = is_learn_new_layers_only

