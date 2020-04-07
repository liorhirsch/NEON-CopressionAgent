import torch
from sklearn.metrics import mean_squared_error

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
        # TODO .to(device) add support for device
        validation_output = self.model(torch.Tensor(self.cross_validation_obj.x_test).float()).reshape(
            -1).detach().cpu().numpy()
        return mean_squared_error(validation_output, self.cross_validation_obj.y_test)

    def train_model(self):
        dataSet = Dataset(self.cross_validation_obj.x_train, self.cross_validation_obj.y_train)
        trainLoader = torch.utils.data.DataLoader(dataSet, batch_size=32, shuffle=True)

        for epoch in range(50):
            for i, batch in enumerate(trainLoader, 0):
                curr_x, curr_y = batch
                self.optimizer.zero_grad()
                # TODO add support for curr_x.to(device).
                outputs = self.model(curr_x.float()).reshape(-1)
                # TODO add support for curr_y.to(device).
                loss = self.loss_func(outputs, curr_y.float())
                loss.backward()
                self.optimizer.step()


