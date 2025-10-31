import torch
import torch.nn as nn
from config import PATCH_SIZE, MARGIN

# -----------------------------
# Siamese model
# -----------------------------

class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim: int = 10):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, padding=2)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, padding=2)
        self.pool2 = nn.AvgPool2d(2)
        # Flatten and project with LazyLinear so weights are created on the fly
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(embed_dim)   # initializes on first forward, on the correct device

    def forward(self, x):
        x = self.bn0(x)
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.tanh(self.fc(x))
        return x

class SiameseNet(nn.Module):
    def __init__(self, patch_size: int = PATCH_SIZE, embed_dim: int = 10):
        super().__init__()
        self.embed = EmbeddingNet(embed_dim)
        self.bn = nn.BatchNorm1d(1)
        self.out = nn.Linear(1, 1)

    def forward(self, a, b):
        va = self.embed(a)
        vb = self.embed(b)
        dist = torch.sqrt(torch.clamp(torch.sum((va - vb) ** 2, dim=1, keepdim=True), min=1e-12))
        dist = self.bn(dist)
        logits = self.out(dist)
        return torch.sigmoid(logits)  # probability in [0,1]

def contrastive_loss(margin: float = MARGIN):
    def _loss(y_true, y_pred):
        # y_true: 0 for similar (positive), 1 for dissimilar (negative)
        y_true = y_true.to(y_pred.dtype)
        pos = (y_pred ** 2)
        neg = torch.clamp(margin - y_pred, min=0.0) ** 2
        return torch.mean((1.0 - y_true) * pos + y_true * neg)
    return _loss
