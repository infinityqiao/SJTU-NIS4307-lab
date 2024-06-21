import torch
from PIL import Image
import torchvision.transforms as transforms
from model import ViolenceClassifier
import os
from sklearn.metrics import accuracy_score, f1_score

class ViolenceClass:
    def __init__(self, ckpt_path, device='cuda:0'):
        """
        初始化函数，加载模型和设置参数
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
        self.model.to(self.device)
        self.model.eval()  # 设置模型为评估模式
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # 保持图像尺寸不变
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def misc(self, img_paths: list) -> torch.Tensor:
        """
        图像预处理函数，按批处理图像
        """
        images = []
        for img_path in img_paths:
            img = self.preprocess(Image.open(img_path).convert('RGB'))
            images.append(img)
            if len(images) == 10:  # 达到一个小批量就处理，这里批量大小设为10
                yield torch.stack(images)
                images = []
        if images:  # 处理剩余的图像
            yield torch.stack(images)

    def classify(self, imgs: torch.Tensor) -> list:
        """
        图像分类函数
        """
        imgs = imgs.to(self.device)
        with torch.no_grad():
            outputs = self.model(imgs)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().tolist()

