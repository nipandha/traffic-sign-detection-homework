import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, kernel_size=7)
        self.batch_norm1=nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=4)
        self.batch_norm2=nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=4)
        self.batch_norm3=nn.BatchNorm2d(250)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2250, 300)
        self.batch_norm4=nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300, nclasses)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        x = self.batch_norm1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.batch_norm2(F.relu(F.max_pool2d(self.conv2(x), 2)))
        x = self.batch_norm3(F.relu(F.max_pool2d(self.conv_drop(self.conv3(x)), 2)))
        x = x.view(-1, 2250)
        x = self.batch_norm4(F.relu(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

