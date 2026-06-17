import time
import torch
from pathlib import Path

from angio_gen.utils.anonimizer import benchmark_transformations_from_images, benchmark_transformations_from_masks, choose_image_paths
from angio_gen.utils.config import load_weight_config
from angio_gen.utils.simple_anonimizers import *
from angio_gen.utils.anonimizer import make_anonimizer, make_anon_dataset_from_masks
from angio_gen.models.cgan import CGANModule
from angio_gen.models.diffusion import DiffusionModule



if __name__ == "__main__":
    weights = load_weight_config()["weights"]
    img_paths = choose_image_paths("aia_dataset", num_samples=100)
    
    benchmark_transformations_from_images(LocalContrastNormAnonimizer(), img_paths)
    benchmark_transformations_from_images(BorderCropAnonimizer(), img_paths)
    benchmark_transformations_from_images(LaplaceNoiseAnonimizer(), img_paths)
    benchmark_transformations_from_images(MedianFilterAnonimizer(), img_paths)
    benchmark_transformations_from_images(QuantizationAnonimizer(), img_paths)
    benchmark_transformations_from_images(JPEGCompressionAnonimizer(), img_paths)

    cgan = make_anonimizer(CGANModule, weights["pix2pix"])
    device = cgan.device
    cgan.to(device)
    cgan.eval()
    benchmark_transformations_from_masks(cgan, img_paths, "Pix2Pix", device=device)

    diffusion = make_anonimizer(DiffusionModule, weights["diffusion"])
    device = diffusion.device
    diffusion.to(device)
    diffusion.eval()
    benchmark_transformations_from_masks(diffusion.sample, img_paths, "Diffusion", device=device)