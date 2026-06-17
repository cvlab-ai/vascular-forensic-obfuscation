import torch
import cv2
import torch.nn.functional as F
import numpy as np

from io import BytesIO
from PIL import Image

class BorderCropAnonimizer:
    def __init__(self, crop_percent=0.1):
        self.p = crop_percent

    def __call__(self, x):
        _, _, h, w = x.shape
        ch, cw = int(h * self.p), int(w * self.p)
        mask = torch.ones_like(x)
        mask[:, :, :ch, :] = 0
        mask[:, :, -ch:, :] = 0
        mask[:, :, :, :cw] = 0
        mask[:, :, :, -cw:] = 0
        return x * mask

class ContrastNormAnonimizer:
    def __call__(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)


class LocalContrastNormAnonimizer:
    def __init__(self, kernel_size=15, sigma=3.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x):
        if x.dim() == 3: x = x.unsqueeze(0)
        coords = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        g = torch.exp(-(coords**2) / (2 * self.sigma**2))
        g = g / g.sum()
        kernel = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
        kernel = kernel.to(x.device)
        mu = F.conv2d(x, kernel, padding=self.kernel_size // 2)
        sigma = torch.sqrt(F.conv2d((x - mu)**2, kernel, padding=self.kernel_size // 2) + 1e-8)
        x_norm = (x - mu) / sigma
        return torch.clamp(x_norm, -1.0, 1.0)


class LaplaceNoiseAnonimizer:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, x):
        dist = torch.distributions.laplace.Laplace(0, self.epsilon)
        noise = dist.sample(x.shape).to(x.device)
        return torch.clamp(x + noise, -1.0, 1.0)


class MedianFilterAnonimizer:
    def __init__(self, kernel_size=5):
        self.k = kernel_size

    def __call__(self, x):
        x_np = ((x.detach().cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        for c in range(x_np.shape[0]):
            x_np[c] = cv2.medianBlur(x_np[c], self.k)
            
        x_out = torch.from_numpy(x_np).to(x.device).float()
        return (x_out / 255.0) * 2.0 - 1.0

class QuantizationAnonimizer:
    def __init__(self, levels=8):
        self.levels = levels

    def __call__(self, x):
        x_norm = (x + 1.0) / 2.0
        x_quant = torch.round(x_norm * (self.levels - 1)) / (self.levels - 1)
        return x_quant * 2.0 - 1.0


class JPEGCompressionAnonimizer:
    def __init__(self, quality=20):
        self.quality = quality

    def __call__(self, x):
        x_np = ((x.detach().cpu().numpy().squeeze() + 1.0) / 2.0 * 255).astype(np.uint8)
        pil_img = Image.fromarray(x_np)
        
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        x_comp = np.array(compressed_img).astype(np.float32)
        x_out = torch.from_numpy(x_comp).to(x.device).unsqueeze(0)
        return (x_out / 255.0) * 2.0 - 1.0
