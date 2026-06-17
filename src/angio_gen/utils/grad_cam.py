import numpy as np
import torch
import torch.nn.functional as F


class HospitalClassifierGradCAM:
    def __init__(self, model, layer_id=11):
        self.model = model
        self.gradients = None
        self.activations = None
        
        self.target_layer = self.model.features[layer_id]
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, img_tensors, classes):
        self.model.eval()
        img_tensors = img_tensors.to(self.model.device)
        classes = classes.to(self.model.device)
        logits = self.model(img_tensors)
        
        scores = logits[torch.arange(logits.size(0)), classes]
        
        retain_graph = scores.shape[0] > 1

        self.model.zero_grad()
        gradients_list = []
        for i, score in enumerate(scores):
            score.backward(retain_graph=retain_graph)
            gradients_list.append(self.gradients[i])

        gradients = torch.stack(gradients_list, dim=0)

        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        heatmap = torch.sum(weights * self.activations, dim=1)
        heatmap = F.relu(heatmap)

        heatmap = heatmap.unsqueeze(1)
        heatmap = F.interpolate(heatmap, size=img_tensors.shape[2:], mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze(1)

        batch_heatmaps = []
        for i in range(heatmap.size(0)):
            img_heatmap = heatmap[i].detach().cpu().numpy()
            if np.max(img_heatmap) != 0:
                img_heatmap = (img_heatmap - np.min(img_heatmap)) / (np.max(img_heatmap) - np.min(img_heatmap))
            batch_heatmaps.append((img_heatmap * 255).astype(np.uint8))

        return batch_heatmaps
    
    