import torch
from sklearn.metrics import accuracy_score

from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.BasicHandler import BasicHandler


class Dataset(torch.utils.data.Dataset):
    def int_to_onehot(self, indx):
        one_hot = torch.zeros(self.range_y).float()
        one_hot[int(indx)] = 1.0
        return one_hot

    def __init__(self, x, y):
        min_y = min(y)
        max_y = max(y)
        self.range_y = int(max_y - min_y + 1)

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.int_to_onehot(self.y[idx])


class ClassificationHandler(BasicHandler):
    def evaluate_model(self) -> float:
        device = StaticConf.getInstance().conf_values.device
        validation_output = torch.argmax(
            self.model(torch.Tensor(self.cross_validation_obj.x_test).to(device).float()).detach().cpu(), 1)
        return accuracy_score(validation_output, self.cross_validation_obj.y_test)

    def train_model(self):

        dataSet = Dataset(self.cross_validation_obj.x_train, self.cross_validation_obj.y_train)
        trainLoader = torch.utils.data.DataLoader(dataSet, batch_size=32, shuffle=True)
        net = self.model.float()
        device = StaticConf.getInstance().conf_values.device

        for epoch in range(StaticConf.getInstance().conf_values.num_epoch):
            running_loss = 0.0

            for i, batch in enumerate(trainLoader, 0):
                curr_x, curr_y = batch

                if len(curr_x) > 1:
                    self.optimizer.zero_grad()
                    outputs = net(curr_x.float().to(device))
                    curr_y = torch.max(curr_y, 1)[1]
                    loss = self.loss_func(outputs, curr_y.to(device))
                    loss.backward()
                    self.optimizer.step()
