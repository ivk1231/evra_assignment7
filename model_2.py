import torch.nn as nn
import torch.nn.functional as F

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        # Optimized model with batch normalization and minimal dropout
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.dropout = nn.Dropout(0.1)  # Further reduced dropout
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x 