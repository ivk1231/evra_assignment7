import torch.nn as nn
import torch.nn.functional as F

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # 8 filters
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # 16 filters
        self.fc1 = nn.Linear(16 * 7 * 7, 64)  # Fully connected layer
        self.fc2 = nn.Linear(64, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        # Define a different architecture
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)  # 10 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1) # 20 filters
        self.fc1 = nn.Linear(20 * 7 * 7, 50)  # Fully connected layer
        self.fc2 = nn.Linear(50, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 20 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 