import torch.nn as nn

class HybridCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(HybridCNN, self).__init__()
        # 28×28 입력 → (1)→ [Conv(64)→Bn→ReLU→Conv(64)→Bn→ReLU→MaxPool] → [Conv(128)→Bn→ReLU→Conv(128)→Bn→ReLU→MaxPool]
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 출력: 64 ×14×14

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 출력: 128×7×7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),               # 128×7×7 = 6272
            nn.Dropout(p=0.5),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.float()
        x = self.features(x)
        x = self.classifier(x)
        return x
