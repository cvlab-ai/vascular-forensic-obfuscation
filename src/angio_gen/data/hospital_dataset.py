import torch
import json
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image
from pathlib import Path

class HospitalDataset(Dataset):
    def __init__(self, root_dir, metadata_list, transform=None, label_map=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label_map = label_map
        
        existing_indices = {f.stem.replace('_', '') for f in self.root_dir.glob("_[0-9]*.png") 
                            if ".mask" not in f.name}
        self.samples = [m for m in metadata_list if str(m['index']) in existing_indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_meta = self.samples[idx]
        img_path = self.root_dir / f"_{sample_meta['index']}.png"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_map[sample_meta['hospital_id']]
        
        return image, torch.tensor(label, dtype=torch.long)


class HospitalDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, metadata_path, batch_size=32, img_size=224):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.metadata_path = Path(metadata_path)
        self.batch_size = batch_size
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
            
        unique_hospitals = sorted(list(set(m['hospital_id'] for m in metadata)))
        self.label_map = {name: i for i, name in enumerate(unique_hospitals)}
        self.num_classes = len(unique_hospitals)

        self.train_ds = HospitalDataset(self.dataset_dir / "train", metadata, 
                                        transform=self.transform, label_map=self.label_map)
        self.val_ds = HospitalDataset(self.dataset_dir / "val", metadata, 
                                        transform=self.transform, label_map=self.label_map)
        self.test_ds = HospitalDataset(self.dataset_dir / "test", metadata, 
                                           transform=self.transform, label_map=self.label_map)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4)
    
    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4)