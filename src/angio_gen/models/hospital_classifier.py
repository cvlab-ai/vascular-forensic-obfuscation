import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy


class HospitalClassifier(pl.LightningModule):
    def __init__(self, num_classes=7, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes

        p_ch = 16

        self.features = nn.Sequential(
            nn.Conv2d(3, p_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(p_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(p_ch, p_ch * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(p_ch * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(p_ch * 2, p_ch * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(p_ch * 4),
            nn.ReLU(),

            nn.Conv2d(p_ch * 4, p_ch * 8, kernel_size=3, padding=1),
            nn.InstanceNorm2d(p_ch * 8),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)) 
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(p_ch * 8, num_classes)
        )

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_f1(logits, y)
        self.log("loss/train_loss", loss, prog_bar=True)
        self.log("f1/train_f1", self.train_f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_f1(logits, y)
        self.log("loss/val_loss", loss, prog_bar=True)
        self.log("f1/val_f1", self.val_f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                # "monitor": "val_loss",
            }
        }
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_f1(logits, y)
        self.log("f1/test_f1", self.test_f1, prog_bar=True)
