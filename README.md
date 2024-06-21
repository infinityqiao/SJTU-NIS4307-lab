# 暴力图片检测

1. `12-modeule`文件夹中有如下文件
   - `dateset.py`：自定义的数据集类和数据模块类
   - `test.py`：模型测试文件
   - `train.py`：模型训练文件
2. `12-classify`.py：接口文件
3. `12-model`.py：模型定义文件
4. `12-noise_generate.py`：为数据集添加扰动，生成对抗样本
5. `12-requirements.txt`：环境配置
6. `12-resnet50_pretrain_test-epoch=18-val_loss=0.11.ckpt`：模型权重保存

使用示例：

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
from model_new import ViolenceClassifier
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
    
# 使用示例
# 假设你的测试图像路径列表是 /root/test_image/ 中的所有文件
test_image_paths = [f"/root/test_image/{i}" for i in os.listdir("/root/test_image")]

# 假设模型保存在/root/12-resnet50_pretrain_test-epoch=18-val_loss=0.11.ckpt
ckpt_path = "/root/12-resnet50_pretrain_test-epoch=18-val_loss=0.11.ckpt"
classifier = ViolenceClass(ckpt_path)

# 存储所有预测结果
all_predictions = []

# 调用 misc 方法进行预处理并分类
for test_images in classifier.misc(test_image_paths):
    predictions = classifier.classify(test_images)
    all_predictions.extend(predictions)

true_labels = [0 if os.path.basename(path).startswith('0_') else 1 for path in test_image_paths]

# 计算准确率和F1指标
accuracy = accuracy_score(true_labels, all_predictions)
f1 = f1_score(true_labels, all_predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
```

