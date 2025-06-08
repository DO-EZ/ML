from typing import Tuple
import os

import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms

from user_dataset import UserImageDataset

def get_mnist_only_dataloaders(
    batch_size: int = 64,
    mnist_root: str = "data"
) -> Tuple[DataLoader, DataLoader]:
    """
    MNIST 데이터만 사용하는 DataLoader를 반환.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST(root=mnist_root, train=True, download=True, transform=transform)
    mnist_test  = datasets.MNIST(root=mnist_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, test_loader


def get_combined_dataloaders(
    batch_size: int = 64,
    user_images_dir: str = "user_data/images",
    user_labels_csv: str  = "user_data/images/labels.csv",
    mnist_root: str = "data"
) -> Tuple[DataLoader, DataLoader]:
    """
    1) MNIST train/test 데이터를 불러오고
    2) user_images_dir에 있는 이미지와 labels.csv로 UserImageDataset 생성
    3) ConcatDataset + WeightedRandomSampler로 7:3 비율로 섞인 train_loader 생성
    4) MNIST test만으로 구성된 test_loader 생성
    """
    # 1) MNIST dataset 로드
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST(
        root=mnist_root, train=True, download=True, transform=transform_mnist
    )
    mnist_test  = datasets.MNIST(
        root=mnist_root, train=False, download=True, transform=transform_mnist
    )

    # 2) User dataset 경로/라벨 확인
    if not os.path.isdir(user_images_dir):
        raise FileNotFoundError(f"유저 이미지 디렉토리가 없습니다: {user_images_dir}")
    if not os.path.isfile(user_labels_csv):
        raise FileNotFoundError(f"라벨 CSV 파일이 없습니다: {user_labels_csv}")

    # 3) UserImageDataset 생성
    user_dataset = UserImageDataset(
        images_dir=user_images_dir,
        labels_csv=user_labels_csv
    )

    # 4) ConcatDataset + WeightedRandomSampler (MNIST 70%, User 30%)
    combined_dataset = ConcatDataset([mnist_train, user_dataset])
    num_mnist = len(mnist_train)
    num_user  = len(user_dataset)
    total_len = num_mnist + num_user

    # 각 샘플에 부여할 weight 계산
    weights = [0.7 / num_mnist] * num_mnist + [0.3 / num_user] * num_user

    # 디버그: weight 합 확인
    total_weight_mnist = sum(weights[:num_mnist])
    total_weight_user = sum(weights[num_mnist:])
    print(f"[DataLoader] sampling weights sum: MNIST={total_weight_mnist:.2f}, User={total_weight_user:.2f}")

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=total_len,
        replacement=True
    )
    indices = list(sampler)
    mnist_count = sum(1 for i in indices if i < num_mnist)
    user_count  = sum(1 for i in indices if i >= num_mnist)
    print(f"[DEBUG] 실제 샘플링 비율 → MNIST: {mnist_count/len(indices):.2f}, User: {user_count/len(indices):.2f}")
    
    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # 5) MNIST test 전용 DataLoader
    test_loader = DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader
