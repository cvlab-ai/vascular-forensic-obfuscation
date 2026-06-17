import random
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def prepare_ICANJ(icanj_dir, output_root):
    IMAGES_DIR = icanj_dir / "V1.0" / "ICA_PNG"
    MASK_DIR = icanj_dir / "V1.0" / "label"

    if not icanj_dir.exists():
        try:
            print(f"Cloning repository to {icanj_dir}...")
            subprocess.run([
                "git", "clone",
                "https://github.com/laudominik/ICA_NJ_BinarySeg.git",
                str(icanj_dir)
            ], check=True)
        except Exception as e:
            print(f"Clone failed: {e}")
            return

    all_valid_images = []
    for img_path in IMAGES_DIR.glob("*.png"):
        if (MASK_DIR / img_path.name).exists():
            all_valid_images.append(img_path)

    print(f"ICA_NJ: Found {len(all_valid_images)} valid pairs.")
    
    train_imgs, temp_imgs = train_test_split(all_valid_images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    splits = {
        "training": train_imgs,
        "validation": val_imgs,
        "test": test_imgs
    }

    for split_name, image_list in splits.items():
        img_out = output_root / split_name / "images"
        mask_out = output_root / split_name / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        print(f"Processing {split_name} split ({len(image_list)} images)...")
        
        for img_path in tqdm(image_list):
            mask_path = MASK_DIR / img_path.name
            shutil.copy(img_path, img_out / img_path.name)
            shutil.copy(mask_path, mask_out / img_path.name)

    print(f"\nDone! ICA_NJ dataset created at: {output_root}")

prepare_ICANJ(
    icanj_dir=Path("data/ICA_NJ_Raw"),
    output_root=Path("re_dataset"),
)