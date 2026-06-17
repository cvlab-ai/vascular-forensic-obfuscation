import os
import pytorch_lightning as pl
import copy

from pathlib import Path
from PIL import Image
from torchvision.datasets import VisionDataset

class ICABinaryDataset(VisionDataset):
    def __init__(self, root, _set=None, transform=None, target_transform=None, transforms=None):
        super().__init__(root, transforms, transform, target_transform)
        self.root = Path(root)

        if _set is not None:
            self.input_dir = self.root / _set / 'images'
            self.label_dir = self.root / _set / 'masks'
        else:
            self.input_dir = self.root / 'images'
            self.label_dir = self.root / 'masks'

        if not self.input_dir.is_dir():
            raise RuntimeError(f"Input directory '{self.input_dir}' not found.")
        if not self.label_dir.is_dir():
            raise RuntimeError(f"Label directory '{self.label_dir}' not found.")

        image_filenames = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        ]
        try:
            image_filenames.sort(key=lambda x: int(x.split(".")[0]))
        except:
            print("WARNING: It is not possible to sort masks.")
        self.samples = []

        for filename in image_filenames:
            input_path = self.input_dir / filename
            label_path = self.label_dir / filename

            if label_path.exists():
                self.samples.append((input_path, label_path))
            else:
                print(f"Warning: Corresponding label for {filename} not found at {label_path}. Skipping.")

        print(f"Found {len(self.samples)} image-label pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path, label_path = self.samples[index]

        input_image = Image.open(input_path).convert("L")
        label_mask = Image.open(label_path).convert("L")

        if self.transforms is not None:
            input_image, label_mask = self.transforms(input_image, label_mask)

        return input_image, label_mask

class ICABinaryDatasetTest(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, transforms=None):
        super().__init__(root, transforms, transform, target_transform)
        self.input_dir = Path(root) / "test" / "masks"

        if not self.input_dir.is_dir():
            raise RuntimeError(f"Input directory '{self.input_dir}' not found.")

        self.samples = []
        image_filenames = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        ]
        try:
            image_filenames.sort(key=lambda x: int(x.split(".")[0]))
        except:
            print("WARNING: It is not possible to sort masks.")

        for filename in image_filenames:
            input_path = self.input_dir / filename
            self.samples.append(input_path)

        print(f"Found {len(self.samples)} masks.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path = self.samples[index]

        input_image = Image.open(input_path).convert("L")

        if self.transform is not None:
            input_image = self.transform(input_image)

        return input_image, input_path.name

class BinarySegmentationDataset(VisionDataset):
    def __init__(self, root, split, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)
        self.root = Path(root)
        self.split_dir = self.root / split

        if not self.split_dir.is_dir():
            raise RuntimeError(f"Directory not found: '{self.split_dir}'")

        self.samples = []
        all_files = os.listdir(self.split_dir)
        mask_files = [f for f in all_files if f.endswith('.mask.png')]
        
        for mask_name in mask_files:
            mask_path = self.split_dir / mask_name
            base_id = mask_name.replace('.mask.png', '')
            image_name = f"{base_id}.png"
            image_path = self.split_dir / image_name

            if image_path.exists():
                self.samples.append((image_path, mask_path))

        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path, label_path = self.samples[index]

        input_image = Image.open(input_path).convert("L")
        label_mask = Image.open(label_path).convert("L")

        if self.transforms is not None:
            input_image, label_mask = self.transforms(input_image, label_mask)

        return input_image, label_mask