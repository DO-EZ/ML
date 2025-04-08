import base64
import io
from PIL import Image
import torch
from torchvision import transforms

def preprocess_base64_image(base64_string: str) -> torch.Tensor:
    # 1. base64 디코딩하고 흑백 이미지로 변환
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("L")

    # 2. 28x28 크기로 사이즈 바꾸고 Tensor 변환
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    # 3. 전처리 적용
    tensor = transform(image)

    # 4. (1, 1, 28, 28)
    tensor = tensor.unsqueeze(0)

    return tensor
