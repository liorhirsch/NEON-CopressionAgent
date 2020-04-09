class ConfigurationValues():
    device: str
    num_epoch: int

    def __init__(self, device, num_epoch = 50) -> None:
        self.device = device
        self.num_epoch = num_epoch

