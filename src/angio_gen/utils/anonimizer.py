from ..data.transforms import apply_test_transform
import torch
import os
import torch
import shutil
from pathlib import Path
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import time
import numpy as np



def make_anonimizer(Model, ckpt):
    model = Model.load_from_checkpoint(ckpt)
    model.eval()
    return model


def load_mask(mask_path):
    mask_pil = Image.open(mask_path).convert("L")
    x_mask = apply_test_transform(mask_pil)
    if x_mask.dim() == 2:    # [H, W]
        x_input = x_mask.unsqueeze(0).unsqueeze(0)
    elif x_mask.dim() == 3:  # [C, H, W]
        x_input = x_mask.unsqueeze(0)
    else:
        x_input = x_mask
    return x_input


def denorm(t, to_numpy=True):
    v =  torch.clamp((t.detach().cpu().squeeze() + 1.0) / 2.0, 0, 1)
    return v.numpy() if to_numpy else v

def anonimize_from_PIL(anz, mask_pil):
    x_mask = apply_test_transform(mask_pil)
    if x_mask.dim() == 2:
        x_input = x_mask.unsqueeze(0).unsqueeze(0)
    elif x_mask.dim() == 3:
        x_input = x_mask.unsqueeze(0)
    else:
        x_input = x_mask

    x_input = x_input.to(device)
    with torch.no_grad():
        x_anon = anz(x_input)
    if x_anon.dim() == 4:
        x_anon = x_anon.squeeze(0)
    if x_anon.size(0) == 1:
        x_anon = x_anon.repeat(3, 1, 1)

    def postprocess(x, low_p=5, high_p=95):
        flat = x.view(-1)
        low_val = torch.quantile(flat, low_p / 100.0)
        high_val = torch.quantile(flat, high_p / 100.0)
        x = torch.clamp(x, low_val, high_val)
        x = (x - low_val) / (high_val - low_val + 1e-8) * 2 - 1
        return x
    
    x_anon = postprocess(x_anon)[0, :, :]
    return x_anon


def make_anon_dataset_from_masks(anz_model, input_dir="aia_dataset", output_dir="aia_dataset_anon", device="cuda"):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(exist_ok=True)
    
    for split in ["train", "val", "test"]:
        src_path = input_root / split
        dst_path = output_root / split
        dst_path.mkdir(parents=True, exist_ok=True)
        
        img_files = [f for f in src_path.glob("_[0-9]*.png") if ".mask" not in f.name]
        print(f"Generating anonymized images from masks in {split}...")

        for f in tqdm(img_files):
            mask_path = f.with_suffix(".mask.png")
            if not mask_path.exists():
                mask_path = src_path / f.name.replace(".png", ".mask.png")

            x_input = load_mask(mask_path)
            x_input = x_input.to(device)

            with torch.no_grad():
                x_anon = anz_model(x_input)
                if x_anon.dim() == 4:
                    x_anon = x_anon.squeeze(0)
                if x_anon.size(0) == 1:
                    x_anon = x_anon.repeat(3, 1, 1)
                x_anon = denorm(x_anon, to_numpy=False)
            save_image(x_anon, dst_path / f.name)
            shutil.copy(mask_path, dst_path / mask_path.name)

    if (input_root / "metadata.json").exists():
        shutil.copy(input_root / "metadata.json", output_root / "metadata.json")

def load_image(path, size=(512, 512)):
    img = Image.open(path).convert("L")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0)

def make_anon_dataset_from_images(anz_model, input_dir="aia_dataset", output_dir="aia_dataset_anon"):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(exist_ok=True)
    
    for split in ["train", "val", "test"]:
        src_path = input_root / split
        dst_path = output_root / split
        dst_path.mkdir(parents=True, exist_ok=True)
        
        img_files = [f for f in src_path.glob("_[0-9]*.png") if ".mask" not in f.name]
        print(f"Applying perturbation to images in {split}...")

        for f in tqdm(img_files):
            x_input = load_image(f)

            with torch.no_grad():
                x_anon = anz_model(x_input)
                
                if x_anon.dim() == 4:
                    x_anon = x_anon.squeeze(0)
                if x_anon.size(0) == 1:
                    x_anon = x_anon.repeat(3, 1, 1)
                x_anon = denorm(x_anon, to_numpy=False)

            save_image(x_anon, dst_path / f.name)
            
            mask_name = f.name.replace(".png", ".mask.png")
            mask_path = src_path / mask_name
            if mask_path.exists():
                shutil.copy(mask_path, dst_path / mask_name)

    if (input_root / "metadata.json").exists():
        shutil.copy(input_root / "metadata.json", output_root / "metadata.json")

def choose_image_paths(input_dir="aia_dataset", num_samples=100):
    input_root = Path(input_dir)
    return [f for f in input_root.rglob("_[0-9]*.png") if ".mask" not in f.name][:num_samples]

def benchmark_transformations_from_masks(anz_model, img_paths, name, device="cuda"):
    mask_times = []
    for f in img_paths:
        mask_path = f.parent / f.name.replace(".png", ".mask.png")
        if not mask_path.exists():
            continue
            
        start = time.perf_counter()
        x_input = load_mask(mask_path).to(device)
        with torch.no_grad():
            x_anon = anz_model(x_input)
            if x_anon.dim() == 4:
                x_anon = x_anon.squeeze(0)
            if x_anon.size(0) == 1:
                x_anon = x_anon.repeat(3, 1, 1)
            _ = denorm(x_anon, to_numpy=False)
        mask_times.append(time.perf_counter() - start)

    avg_mask_time = np.mean(mask_times) if mask_times else 0
    std_mask_time = np.std(mask_times) if len(mask_times) > 1 else 0

    print(f"Samples processed: {len(mask_times)} masks")
    print(f"Average Mask Transformation Time for {name}: {avg_mask_time:.4f} ± {std_mask_time:.4f} seconds")
    
def benchmark_transformations_from_images(anz_model, img_paths):
    img_times = []
    for f in img_paths:
        start = time.perf_counter()
        x_input = load_image(f)
        with torch.no_grad():
            x_anon = anz_model(x_input)
            if x_anon.dim() == 4:
                x_anon = x_anon.squeeze(0)
            if x_anon.size(0) == 1:
                x_anon = x_anon.repeat(3, 1, 1)
            _ = denorm(x_anon, to_numpy=False)
        img_times.append(time.perf_counter() - start)

    avg_img_time = np.mean(img_times) if img_times else 0
    std_img_time = np.std(img_times) if len(img_times) > 1 else 0

    print(f"Samples processed: {len(img_times)} images")
    print(f"Average Image Transformation Time for {anz_model.__class__.__name__}: {avg_img_time:.4f} ± {std_img_time:.4f} seconds")