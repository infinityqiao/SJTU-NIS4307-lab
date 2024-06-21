from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from PIL import ImageFilter
import torch
import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
def random_gaussian_blur(img):
    if random.random() < 0.5:
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.25, 1))) # 随机高斯模糊
    return img


class CustomDataset(Dataset):
    def __init__(self, split):
        assert split in ["train", "val", "test"]
        data_root = "/home/huhw/ai-project/violence_dataset/"
        self.data = [os.path.join(data_root, split, i) for i in os.listdir(data_root + split)]
        base_transforms = [
            transforms.Resize((224, 224)),  # 调整图像大小到224x224
            transforms.ToTensor(),  # 将图像转换为Tensor
            AddGaussianNoise(0., 0.05),  # 添加高斯噪声
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
        ]
        if split == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # 随机翻转
                transforms.Lambda(random_gaussian_blur),  # 随机高斯模糊
                *base_transforms
            ])
        else:
            self.transforms = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        try:
            x = Image.open(img_path).convert("RGB")
        except:
            print(img_path)
            return None
        if x is None:
            print("img is None")
            return None
        y = int(img_path.split("/")[-1][0])  # 获取标签值，0代表非暴力，1代表暴力
        x = self.transforms(x)
        return x, y


class CustomDataModule(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # 分割数据集、应用变换等
        # 创建 training, validation数据集
        self.train_dataset = CustomDataset("train")
        self.val_dataset = CustomDataset("val")
        #self.test_dataset = CustomDataset("test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
