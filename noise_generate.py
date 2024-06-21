<<<<<<< HEAD
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image

# 设置输入和输出目录
input_dir = '/home/huhw/ai-project/new_violence/violence'  # 修改为你的实际输入目录
output_dir = '/home/huhw/ai-project/noise'  # 修改为你的实际输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载预训练模型
model = models.resnet18(pretrained=True)
model.eval()

loss_fn = nn.CrossEntropyLoss()

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# FGSM 生成对抗样本
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # 限制对抗样本的像素值在 [0, 1] 之间
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# 遍历输入目录中的所有图像文件
epsilon = 0.1  # 扰动的强度
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        epsilon = 0.1  # 扰动的强度
count = 0 

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        if count < 1000:  # 只处理前1000个图像
            image_path = os.path.join(input_dir, filename)

            # 打开图像并应用预处理
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0)

            # 需要在梯度计算中启用输入图像的梯度
            image.requires_grad = True

            # 获取图像的标签
            label = torch.tensor([0])  # 如果没有标签信息，可以临时设置为 0 或使用其他合适的标签

            # 前向传播并计算损失
            output = model(image)
            loss = loss_fn(output, label)
            model.zero_grad()
            loss.backward()
            data_grad = image.grad.data

            # 生成对抗样本
            perturbed_image = fgsm_attack(image, epsilon, data_grad)

            # 生成新的文件名
            name, ext = os.path.splitext(filename)
            new_filename = f"{name.split('_')[0]}_aei{name.split('_')[1]}{ext}"
            new_path = os.path.join(output_dir, new_filename)

            # 保存对抗样本
            save_image(perturbed_image, new_path)
            print(f"Saved adversarial image as {new_filename}")

            count += 1  
        else:
            break  


print("All adversarial images have been generated and saved.")
=======
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image

# 设置输入和输出目录
input_dir = '/home/huhw/ai-project/new_violence/violence'  # 修改为你的实际输入目录
output_dir = '/home/huhw/ai-project/noise'  # 修改为你的实际输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载预训练模型
model = models.resnet18(pretrained=True)
model.eval()

loss_fn = nn.CrossEntropyLoss()

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# FGSM 生成对抗样本
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # 限制对抗样本的像素值在 [0, 1] 之间
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# 遍历输入目录中的所有图像文件
epsilon = 0.1  # 扰动的强度
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        epsilon = 0.1  # 扰动的强度
count = 0 

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        if count < 1000:  # 只处理前1000个图像
            image_path = os.path.join(input_dir, filename)

            # 打开图像并应用预处理
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0)

            # 需要在梯度计算中启用输入图像的梯度
            image.requires_grad = True

            # 获取图像的标签
            label = torch.tensor([0])  # 如果没有标签信息，可以临时设置为 0 或使用其他合适的标签

            # 前向传播并计算损失
            output = model(image)
            loss = loss_fn(output, label)
            model.zero_grad()
            loss.backward()
            data_grad = image.grad.data

            # 生成对抗样本
            perturbed_image = fgsm_attack(image, epsilon, data_grad)

            # 生成新的文件名
            name, ext = os.path.splitext(filename)
            new_filename = f"{name.split('_')[0]}_aei{name.split('_')[1]}{ext}"
            new_path = os.path.join(output_dir, new_filename)

            # 保存对抗样本
            save_image(perturbed_image, new_path)
            print(f"Saved adversarial image as {new_filename}")

            count += 1  
        else:
            break  


print("All adversarial images have been generated and saved.")
>>>>>>> origin/main
