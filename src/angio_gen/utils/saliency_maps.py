import numpy as np
import torch
import torch.nn.functional as F


class HospitalClassifierSaliencyMaps:
    def __init__(self, model):
        self.model = model

    def generate_heatmap(self, img_tensors, classes):
        self.model.eval()
        img_tensors = img_tensors.to(self.model.device)
        img_tensors.requires_grad_()
        classes = classes.to(self.model.device)
        logits = self.model(img_tensors)
        
        scores = logits[torch.arange(logits.size(0)), classes]
        
        retain_graph = scores.shape[0] > 1

        self.model.zero_grad()
        saliency_maps_list = []
        for i, score in enumerate(scores):
            score.backward(retain_graph=retain_graph)
            saliency, _ = torch.max(img_tensors.grad[i].data.abs(), dim=0)
            saliency_maps_list.append(saliency)

        heatmap = torch.stack(saliency_maps_list, dim=0)

        batch_heatmaps = []
        for i in range(heatmap.size(0)):
            img_heatmap = heatmap[i].detach().cpu().numpy()
            if np.max(img_heatmap) != 0:
                img_heatmap = (img_heatmap - np.min(img_heatmap)) / (np.max(img_heatmap) - np.min(img_heatmap))
            batch_heatmaps.append((img_heatmap * 255).astype(np.uint8))

        return batch_heatmaps