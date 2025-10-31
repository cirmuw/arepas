import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
)

# -----------------------------
# Utility layers
# -----------------------------

class InstanceNormalization(nn.Module):
    """InstanceNorm2d with learnable affine parameters, TF default eps=1e-5."""
    def __init__(self, num_features, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=True, eps=eps)

    def forward(self, x):
        return self.norm(x)


def snconv2d(in_ch, out_ch, k, s=1, p=0, bias=True, n_power_iterations=1):
    """Spectral normalized Conv2d equivalent to TF SNConv2D."""
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
    return nn.utils.spectral_norm(conv, n_power_iterations=n_power_iterations)


# -----------------------------
# Residual block
# -----------------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim: int, p_drop: float = 0.5):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True)
        self.in1 = InstanceNormalization(dim)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True)
        self.in2 = InstanceNormalization(dim)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        identity = x
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.in1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.in2(out)
        return out + identity


# -----------------------------
# Generator (Pix2PixHD-style ResNet)
# -----------------------------
class Pix2PixHDGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, n_res_blocks: int = 9):
        super().__init__()
        layers_ = []

        # c7s1-64
        layers_.append(nn.ReflectionPad2d(3))
        layers_.append(nn.Conv2d(in_channels, 64, kernel_size=7, padding=0, bias=True))
        layers_.append(InstanceNormalization(64))
        layers_.append(nn.ReLU(inplace=True))

        # d128, d256
        layers_.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True))
        layers_.append(InstanceNormalization(128))
        layers_.append(nn.ReLU(inplace=True))

        layers_.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True))
        layers_.append(InstanceNormalization(256))
        layers_.append(nn.ReLU(inplace=True))

        # Residual blocks
        for _ in range(n_res_blocks):
            layers_.append(ResnetBlock(256))

        # u128, u64
        layers_.append(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True))
        layers_.append(InstanceNormalization(128))
        layers_.append(nn.ReLU(inplace=True))

        layers_.append(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True))
        layers_.append(InstanceNormalization(64))
        layers_.append(nn.ReLU(inplace=True))

        # c7s1-out
        layers_.append(nn.ReflectionPad2d(3))
        layers_.append(nn.Conv2d(64, out_channels, kernel_size=7, padding=0, bias=True))

        self.model = nn.Sequential(*layers_)

    def forward(self, x):
        out = self.model(x)
        return torch.tanh(out)


# -----------------------------
# PatchGAN Discriminator (with spectral norm)
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels: int = IMG_CHANNELS):
        super().__init__()
        # input is concatenation of input_img and target_img
        ch = in_channels * 2
        self.c1 = snconv2d(ch,   64, 4, s=2, p=1)
        self.c2 = snconv2d(64,  128, 4, s=2, p=1)
        self.c3 = snconv2d(128, 256, 4, s=2, p=1)
        self.c4 = snconv2d(256, 512, 4, s=1, p=1)
        self.c5 = snconv2d(512,   1, 4, s=1, p=1)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)  # N,C,H,W
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.act(self.c3(x))
        x = self.act(self.c4(x))
        x = self.c5(x)  # logits
        return x


# -----------------------------
# VGG Feature Loss (block3_conv3 ≈ relu3_3)
# -----------------------------
class VGGFeatureExtractor(nn.Module):
    """Outputs features after relu3_3 (index 15 in torchvision VGG19.features)."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        for p in vgg.parameters():
            p.requires_grad = False
        # Up to and including relu3_3 (features index 15). Keras block3_conv3 maps to conv index 14;
        # we include relu for stability.
        self.slice = nn.Sequential(*list(vgg.features.children())[:15])
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.slice(x)


def vgg_loss_fn(vgg_model: VGGFeatureExtractor):
    """Create a callable VGG L2 feature loss for grayscale inputs in [-1,1]."""
    # Normalization for VGG (expects [0,1] then normalized by ImageNet mean/std).
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    @torch.no_grad()
    def normalize_rgb(x):
        return (x - imagenet_mean.to(x.device)) / imagenet_std.to(x.device)

    def _loss(y_true, y_pred):
        # Convert 1-channel [-1,1] -> 3-channel, then to [0,1] for VGG
        def to_rgb_01(x):
            x01 = (x + 1.0) * 0.5
            return x01.repeat(1, 3, 1, 1)

        with torch.no_grad():
            y_true_rgb = to_rgb_01(y_true)
            y_pred_rgb = to_rgb_01(y_pred)
            y_true_rgb = normalize_rgb(y_true_rgb)
            y_pred_rgb = normalize_rgb(y_pred_rgb)
        f_true = vgg_model(y_true_rgb)
        f_pred = vgg_model(y_pred_rgb)
        return torch.mean((f_true - f_pred) ** 2)

    return _loss


# -----------------------------
# Public factory and losses
# -----------------------------
def build_pix2pixhd_generator(input_shape=(1, 256, 256), n_res_blocks=9):
    in_ch = input_shape[0]
    return Pix2PixHDGenerator(in_channels=in_ch, out_channels=1, n_res_blocks=n_res_blocks)

def build_discriminator():
    return Discriminator(in_channels=IMG_CHANNELS)

bce_logits = nn.BCEWithLogitsLoss()

def discriminator_loss(real_out, fake_out):
    real_labels = torch.ones_like(real_out) * 0.9  # label smoothing
    fake_labels = torch.zeros_like(fake_out)
    loss_real = bce_logits(real_out, real_labels)
    loss_fake = bce_logits(fake_out, fake_labels)
    return loss_real + loss_fake

def generator_loss(fake_out):
    target = torch.ones_like(fake_out)
    return bce_logits(fake_out, target)

def gradient_penalty(discriminator, real, fake, input_img):
    """WGAN-GP style penalty on interpolated samples wrt discriminator output."""
    device = real.device
    batch = real.size(0)
    eps = torch.rand(batch, 1, 1, 1, device=device)
    interpolated = eps * real + (1 - eps) * fake
    interpolated.requires_grad_(True)
    d_inter = discriminator(input_img, interpolated)
    grad = torch.autograd.grad(
        outputs=d_inter,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(batch, -1)
    norm = torch.sqrt(torch.sum(grad ** 2, dim=1) + 1e-12)
    return torch.mean((norm - 1.0) ** 2)
