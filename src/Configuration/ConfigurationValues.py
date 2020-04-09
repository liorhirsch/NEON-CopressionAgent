class ConfigurationValues():
    device: str
    num_epoch: int
    num_actions: int

    def __init__(self, device, num_epoch = 50, num_actions = 4) -> None:
        self.device = device
        self.num_epoch = num_epoch
        self.num_actions = num_actions

