import base64
import io
from PIL import Image
import torch
from torchvision import transforms

def preprocess_base64_image(base64_string: str) -> torch.Tensor:
    """
    Base64 문자열로 넘어온 PNG(흑백 숫자 그림)을 디코딩하여
    1×1×28×28 형태의 float32 Tensor로 변환하여 반환.
    Args:
        base64_string (str): "data:image/png;base64,xxxxx..." 와 같은 전체 문자열
    """
    if base64_string.startswith("data:"):
        _, base64_data = base64_string.split(",", 1)
    else:
        base64_data = base64_string

    image_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # 흑백 모드
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor_image = transform(image).unsqueeze(0)  # shape: (1, 1, 28, 28)
    return tensor_image
