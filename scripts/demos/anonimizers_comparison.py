import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms
from angio_gen.utils.config import load_weight_config
from angio_gen.utils.simple_anonimizers import *
from angio_gen.utils.anonimizer import make_anonimizer, load_mask, denorm
from angio_gen.models.cgan import CGANModule
from angio_gen.models.diffusion import DiffusionModule
from angio_gen.data.transforms import apply_test_transform
import matplotlib

matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.port"] = 2137
matplotlib.rcParams['webagg.open_in_browser'] = False


def run_demo(image_path, mask_path, 
            image_anonimizers,
            mask_anonimizers, show_og=True):
    img_pil = Image.open(image_path).convert("L")
    
    to_tensor = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img_t = to_tensor(img_pil).unsqueeze(0)
    mask_t = load_mask(mask_path)
    image_anonimized = []
    mask_anonimized = []


    results = [("Original", denorm(img_t)), ("Mask", denorm(mask_t))] if show_og else []

    with torch.no_grad():
        for title, anz in image_anonimizers:
            results.append((title, denorm(anz(img_t.to(next(anz.parameters()).device if hasattr(anz, 'parameters') else 'cpu')))))
        for title, anz in mask_anonimizers:
            results.append((title, denorm(anz(mask_t.to(next(anz.parameters()).device if hasattr(anz, 'parameters') else 'cpu')))))

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 5))
    if n == 1: axes = [axes]

    for ax, (title, img) in zip(axes, results):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout(pad=2.0)
    plt.show()

if __name__ == "__main__":
    weights = load_weight_config()["weights"]

    pix2pix = make_anonimizer(CGANModule, weights["pix2pix"])
    pix2pix.eval()
    diffusion = make_anonimizer(DiffusionModule, weights["diffusion"])
    diffusion.eval()

    SAMPLE = "aia_dataset/test/_70.png"
    MASK = "aia_dataset/test/_70.mask.png"


    run_demo(SAMPLE, MASK,
            [
                ("CN", LocalContrastNormAnonimizer()),
                ("BC", BorderCropAnonimizer()),
                ("LN (eps=0.1)", LaplaceNoiseAnonimizer()),
            ], [
            ]
    )

    run_demo(SAMPLE, MASK,
            [
                ("MF", MedianFilterAnonimizer()),
                ("QU", QuantizationAnonimizer()),
                ("JPEG", JPEGCompressionAnonimizer())
            ],
            [
                ("Pix2pix", pix2pix),
                ("Diffusion", diffusion.sample)
            ],
            show_og=False
    )
    
