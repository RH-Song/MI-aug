import torch
from torch import nn

num_features = 1200
n_class =  5
dropout = 0.5
class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def cal_flattenNum(num_features, i):
    a = num_features
    for i in range(i):
        a = int((a - 20 + 1) / 2)
    return a

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.CNN = nn.Sequential(
            nn.BatchNorm1d(12),
            nn.Conv1d(in_channels=12, out_channels=64, kernel_size=20),  # PS tensortflow中filters对应outchannel的数目 变成12channel
            nn.BatchNorm1d(64),  # 感觉归一化函数放的位置不对。。。
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),  # tensortflow中kernel_size = 2


            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=20),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),


            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=20),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),


            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=20),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),


            FlattenLayer(),
            nn.Linear(256 * cal_flattenNum(num_features, 4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(2048, n_class),
            nn.Softmax()
        )

    def forward(self, x):  # x shape: (batch, 1, 28, 28)
        y = self.CNN(x)
        return y
