import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = nn.Sequential(
            # 1 x 28 x 28
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 6 x 14 x 14
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)
            # 16 x 5 x 5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        out = self.classifier(x)
        return out