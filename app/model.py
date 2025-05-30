import torch
import torch.nn as nn


class FashionClassifier(nn.Module):
    def __init__(self, num_classes = 10):
        super(FashionClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.ReLU(),
            
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x
    