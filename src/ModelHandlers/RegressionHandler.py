import torch
from sklearn.metrics import mean_squared_error

from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.BasicHandler import BasicHandler


class Dataset(torch.utils.data.Dataset):
    def int_to_onehot(self, indx):
        one_hot = torch.zeros(2).float()
        one_hot[indx] = 1.0
        return one_hot

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class RegressionHandler(BasicHandler):
    def evaluate_model(self) -> float:
        device = StaticConf.getInstance().conf_values.device
        validation_output = self.model(torch.Tensor(self.cross_validation_obj.x_test).to(device).float()).reshape(
            -1).detach().cpu().numpy()
        return mean_squared_error(validation_output, self.cross_validation_obj.y_test)

    def train_model(self):
        dataSet = Dataset(self.cross_validation_obj.x_train, self.cross_validation_obj.y_train)
        trainLoader = torch.utils.data.DataLoader(dataSet, batch_size=32, shuffle=True)
        device = StaticConf.getInstance().conf_values.device

        for epoch in range(StaticConf.getInstance().conf_values.num_epoch):
            for i, batch in enumerate(trainLoader, 0):
                curr_x, curr_y = batch
                self.optimizer.zero_grad()
                outputs = self.model(curr_x.to(device).float()).reshape(-1)
                loss = self.loss_func(outputs, curr_y.to(device).float())
                loss.backward()
                self.optimizer.step()
