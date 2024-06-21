from torch import nn
from torchvision import models
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule
import torch

class ViolenceClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.model = models.resnet50(pretrained=False, num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss() # 交叉熵损失函数
        self.learning_rate = learning_rate 
        self.accuracy = Accuracy(task="multiclass", num_classes=2)
        self.automatic_optimization = False  # 禁用自动优化

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def fgsm_attack(self, data, epsilon, data_grad): # 对抗样本生成
        sign_data_grad = data_grad.sign()
        perturbed_data = data + epsilon * sign_data_grad
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        return perturbed_data

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad = True
        logits = self(x)
        loss = self.loss_fn(logits, y) 
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        
        if torch.rand(1) < 0.10:  # 10%的概率进行对抗训练
            x_grad = x.grad.data
            perturbed_x = self.fgsm_attack(x, epsilon=0.01, data_grad=x_grad)
            perturbed_logits = self(perturbed_x)
            perturbed_loss = self.loss_fn(perturbed_logits, y)
            self.log('train_perturbed_loss', perturbed_loss, on_step=True, on_epoch=True, prog_bar=True) # 记录对抗训练的损失
            optimizer.zero_grad()
            self.manual_backward(perturbed_loss)
            optimizer.step()

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        self.log('test_acc', acc)
        return acc
