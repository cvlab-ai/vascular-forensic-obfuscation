import torch
import torch.nn.functional as F


def compute_energy_distance(x, y):
    def pdist(mat1, mat2):
        return torch.cdist(mat1, mat2, p=2)
    dist_xy = pdist(x, y).mean()
    dist_xx = pdist(x, x).mean()
    dist_yy = pdist(y, y).mean()
    return 2 * dist_xy - dist_xx - dist_yy

def compute_classwise_energy_distance(datamodule):
    datamodule.setup("test")
    loader = datamodule.test_dataloader()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x_flat = x.view(x.size(0), -1) 
            all_features.append(x_flat)
            all_labels.append(y)
            
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    unique_hospitals = torch.unique(all_labels)
    results = {}
    Q_ref = all_features 
    
    for hosp_idx in unique_hospitals:
        P_hosp = all_features[all_labels == hosp_idx]
        e_dist = compute_energy_distance(P_hosp, Q_ref)
        results[f"Hospital_{hosp_idx.item()}"] = e_dist.item()
        
    avg_val = sum(results.values()) / len(results)
    results["Average_EnergyDist"] = avg_val
    return results


def compute_ssim(pred, gt, window_size=11, size_average=True):
    c1, c2 = 0.01**2, 0.03**2
    
    mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu_g = F.avg_pool2d(gt, window_size, stride=1, padding=window_size//2)
    # V{X} = E{X^2} - E{X}^2
    sigma_p = F.avg_pool2d(pred**2, window_size, stride=1, padding=window_size//2) - mu_p**2
    sigma_g = F.avg_pool2d(gt**2, window_size, stride=1, padding=window_size//2) - mu_g**2
    sigma_pg = F.avg_pool2d(pred * gt, window_size, stride=1, padding=window_size//2) - mu_p * mu_g
    cs_map = (2 * sigma_pg + c2) / (sigma_p + sigma_g + c2)
    l_map = (2 * mu_p * mu_g + c1) / (mu_p**2 + mu_g**2 + c1)
    ssim_map = l_map * cs_map
    return ssim_map.mean() if size_average else ssim_map, cs_map


def compute_re_dice(pred_mask, gt_mask, smooth=1e-7):
    pred_flat = pred_mask.view(pred_mask.size(0), -1)
    gt_flat = gt_mask.view(gt_mask.size(0), -1)
    intersection = (pred_flat * gt_flat).sum(1)
    return ((2. * intersection + smooth) / (pred_flat.sum(1) + gt_flat.sum(1) + smooth)).mean()


def soft_skeletonize(x, iters=3):
    for _ in range(iters):
        mag = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = x - F.relu(x - F.avg_pool2d(mag, kernel_size=3, stride=1, padding=1))
    return x


def compute_re_cldice(pred_mask, gt_mask, iters=3, smooth=1e-7):
 
    t_p = soft_skeletonize(pred_mask, iters=iters)
    t_l = soft_skeletonize(gt_mask, iters=iters)
    
    s_p = (t_p * gt_mask).sum() / (t_p.sum() + smooth)
    s_l = (t_l * pred_mask).sum() / (t_l.sum() + smooth)
    
    return (2.0 * s_p * s_l) / (s_p + s_l + smooth)
