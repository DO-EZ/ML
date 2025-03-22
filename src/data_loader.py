from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
