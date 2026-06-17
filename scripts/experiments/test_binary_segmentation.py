from pathlib import Path

import torch
from torch.utils.data import DataLoader

from angio_gen.data.dataset import BinarySegmentationDataset
from angio_gen.utils.config import load_weight_config
from binseg.model.binary_segmentation import BinarySegmentationTrainer
from binseg.data.preprocess import get_transforms

datasets = [
    "aia_dataset",
    "aia_dataset_anon_pix2pix",
    "aia_dataset_anon_cn",
    "aia_dataset_anon_bc",
    "aia_dataset_anon_ln",
    "aia_dataset_anon_jpeg",
    "aia_dataset_anon_qu",
    "aia_dataset_anon_mf",
    "aia_dataset_anon_diffusion"
]

def calculate_binary_segmentation_metric(root, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BinarySegmentationTrainer.load_from_checkpoint(
        load_weight_config()["weights"]["binary_segmentation"],
        lr=1e-4,
        map_location=device
    )
    model.eval()
    all_dice = []
    smooth = 1e-6

    dataset = BinarySegmentationDataset(root, "test", transforms=get_transforms(train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(loader):
            image = image.to(device)
            mask = mask.to(device)

            logits = model(image)
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()

            intersection = (preds * mask).sum(dim=(1, 2, 3))
            total = preds.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
            
            dice = (2. * intersection + smooth) / (total + smooth)
            
            all_dice.extend(dice.cpu().tolist())
    
    avg_f1_dice = sum(all_dice) / len(all_dice)
    
    return avg_f1_dice


for dataset in datasets:
    repo_root = Path(__file__).resolve().parents[2]
    print(f'{dataset}: {calculate_binary_segmentation_metric(str(repo_root / dataset))}')