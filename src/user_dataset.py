import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class UserImageDataset(Dataset):
    """
    사용자 캡차 이미지를 불러오고 레이블을 정수로 변환한 Dataset

    images_dir: 이미지 폴더
    labels_csv: labels.csv 파일 경로 (절대 혹은 images_dir 내 상대)
    """
    def __init__(self, images_dir: str, labels_csv: str = "labels.csv"):
        self.images_dir = images_dir
        # labels_csv가 절대 경로로 존재하면 사용, 아니면 images_dir과 조합
        if os.path.isfile(labels_csv):
            labels_path = labels_csv
        else:
            labels_path = os.path.join(images_dir, labels_csv)

        df = pd.read_csv(labels_path)
        # NaN 레이블 제거
        df = df.dropna(subset=["label"])
        # 레이블을 정수형으로 변환
        df["label"] = df["label"].astype(int)

        self.image_files = df["filename"].tolist()
        self.labels = df["label"].tolist()

        # 이미지 전처리: 28×28 그레이스케일 → Tensor
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label
