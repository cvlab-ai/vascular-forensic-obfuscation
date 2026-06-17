import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from angio_gen.data.hospital_dataset import HospitalDataModule
from angio_gen.models.hospital_classifier import HospitalClassifier
from angio_gen.utils.config import load_weight_config
from angio_gen.utils.saliency_maps import HospitalClassifierSaliencyMaps


def save_saliency_maps_results(model, datamodule, saliency_maps, output_root):
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    model.eval()
    device = model.device

    class_names = getattr(datamodule, 'classes', [f"class_{i}" for i in range(model.num_classes)])

    print(f"Heatmaps generating to directory: {output_root}...")
    
    global_idx = 0
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
        
        heatmaps = saliency_maps.generate_heatmap(images, labels)
        
        for i in range(images.size(0)):
            img_tensor = images[i]
            label = labels[i].item()
            pred = preds[i].item()
            heatmap_np = heatmaps[i]
            
            status = "correctly" if label == pred else "incorrectly"
            label_name = class_names[label]
            
            target_dir = os.path.join(output_root, status, label_name)
            os.makedirs(target_dir, exist_ok=True)
            
            orig_img = img_tensor.permute(1, 2, 0).cpu().detach().numpy()
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min()) * 255
            orig_img = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)
            
            filename = f"img_{global_idx}_true_{label}_pred_{pred}.png"
            cv2.imwrite(os.path.join(target_dir, filename), overlay)
            
            global_idx += 1

if __name__ == "__main__":
    datamodule = HospitalDataModule(
        dataset_dir="aia_dataset", 
        metadata_path=os.path.join("aia_dataset", "metadata.json"), 
        batch_size=1
    )
    datamodule_anon = HospitalDataModule(
        dataset_dir="aia_dataset_anon", 
        metadata_path=os.path.join("aia_dataset_anon", "metadata.json"), 
        batch_size=1
    )
    
    model = HospitalClassifier.load_from_checkpoint(load_weight_config()["weights"]["classifier"])
    gradcam = HospitalClassifierSaliencyMaps(model)
    
    save_saliency_maps_results(model, datamodule, gradcam, f"hospital_clf_saliency_maps")
    save_saliency_maps_results(model, datamodule_anon, gradcam, f"hospital_clf_anon_saliency_maps")
    print("Done.")