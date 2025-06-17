import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import base64
from io import BytesIO
from PIL import Image
import torch
from src.preprocess import preprocess_base64_image

class TestPreprocessAPIImage(unittest.TestCase):
    def setUp(self):
        # 가상의 200x200 픽셀 흑백 이미지 생성
        image = Image.new("L", (200, 200), color=128)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        self.base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

    def test_preprocess_base64_image(self):
        tensor = preprocess_base64_image(self.base64_string)
        # 모델이 기대하는 입력 형태: (1, 1, 28, 28)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 1, 28, 28))

if __name__ == '__main__':
    unittest.main()
