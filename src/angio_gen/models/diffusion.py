import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as utils
import torchvision
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from diffusers.optimization import get_cosine_schedule_with_warmup

from pytorch_lightning import LightningModule
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from tqdm import tqdm

IMAGE_SIZE = 256

def prepare_for_lpips(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    
    return torch.clamp(x, -1.0, 1.0)

class DiffusionModule(LightningModule):
    SAMPLE_TIMESTEPS = 50
    UNET2D_CONFIG = {
        "sample_size": IMAGE_SIZE,
        "in_channels": 2,
        "out_channels": 1,
        "layers_per_block": 2,
        "block_out_channels": (64, 128, 256, 512),
        "down_block_types": ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")      
    }
    
    def __init__(self, lr=1e-4, weight_decay=1e-4, image_size=IMAGE_SIZE, timesteps=1000):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = UNet2DModel(**DiffusionModule.UNET2D_CONFIG)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True
        )
        self.fast_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        self.fast_scheduler.set_timesteps(DiffusionModule.SAMPLE_TIMESTEPS)
        self._last_viz_batch = None
        self.log_interval = 20
        
        self.train_lpips = LPIPS()
        self.val_lpips = LPIPS()
        self.test_lpips = LPIPS()

        self.fid = FID()
    
    def forward(self, image, t, mask, drop_proba=0.1):
        rand = torch.rand(1, device=self.device).item()
        cond = torch.zeros_like(mask, device=self.device) if rand < drop_proba else mask
        inp = torch.cat([image, cond], dim=1)
        return self.model(inp, t).sample

    def any_step(self, batch, batch_idx, mode):
        images, masks = batch
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=self.device).long()
        noisy_x = self.noise_scheduler.add_noise(images, noise, timesteps)
        noise_pred = self(noisy_x, timesteps, masks)
 
        loss = F.mse_loss(noise_pred, noise)
        self.log(f"loss/{mode}", loss, on_epoch=True, on_step=False, prog_bar=True)

        if self._last_viz_batch is None or (batch_idx % self.trainer.log_every_n_steps == 0):
            self._last_viz_batch = (images[:4].detach().cpu(), masks[:4].detach().cpu())

        if self.current_epoch % self.log_interval == 0:
            with torch.no_grad():
                fake_images = self.sample(masks, num_samples=images.shape[0])
            if mode == "train":
                lpips = self.train_lpips
            elif mode == "val":
                lpips = self.val_lpips
            else:
                lpips = self.test_lpips
            lpips.update(prepare_for_lpips(fake_images), prepare_for_lpips(images))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.any_step(batch, batch_idx, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.any_step(batch, batch_idx, "val")
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.any_step(batch, batch_idx, "test")
        return loss
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 2:
            masks, filenames = batch
            result = self.sample(masks, len(masks))
            return {"image_tensor": (result + 1) / 2, "filename": filenames}
        else:
            result = self.sample(batch, len(batch))
            return (result + 1) / 2

    def on_train_epoch_end(self):
        self.log_metrics_and_images("train")

    def on_validation_epoch_end(self):
        self.log_metrics_and_images("val")
    
    def on_test_epoch_end(self):
        self.log_metrics_and_images("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.05)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    @torch.no_grad()
    def sample(self, mask, num_samples=1, guidance_scale=2):
        self.eval()
        device = self.device
        uncond_mask = torch.zeros_like(mask, device=device)
        cond_model_input = torch.cat([uncond_mask, mask], dim=0)
        x = torch.randn((num_samples, 1, IMAGE_SIZE, IMAGE_SIZE), device=device)
        self.fast_scheduler.set_timesteps(self.SAMPLE_TIMESTEPS)
        timesteps = self.fast_scheduler.timesteps
        x = x * self.fast_scheduler.init_noise_sigma
        for i, t in enumerate(tqdm(timesteps, desc=f"Sampling ({self.SAMPLE_TIMESTEPS} steps)...")):
            latent_model_input = torch.cat([x] * 2, dim=0)
            latent_model_input = self.fast_scheduler.scale_model_input(latent_model_input, t)
            t_tensor = torch.full((2 * num_samples,), t, device=device, dtype=torch.long)        
            noise_pred = self(latent_model_input, t_tensor, cond_model_input)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            x = self.fast_scheduler.step(noise_pred, t, x).prev_sample
        return x

    def visualize_predictions(self, input_masks, fake_images, real_images, mode):
        
        pil_real_images = [to_pil_image((image + 1) / 2) for image in real_images]
        pil_input_masks = [to_pil_image((mask + 1) / 2) for mask in input_masks]
        pil_fake_images = [to_pil_image((torch.clamp(image, -1, 1) + 1) / 2) for image in fake_images]

        grid = utils.make_grid([pil_to_tensor(image) for image in (pil_input_masks + pil_real_images + pil_fake_images)])
        self.logger.experiment.add_image(f"{mode}_predictions/images", grid, self.current_epoch)

    def log_metrics_and_images(self, mode):
        if self.current_epoch % self.log_interval == 0:
            if self._last_viz_batch is not None:
                real_images, input_masks = self._last_viz_batch
                input_masks = input_masks.to(self.device)
                real_images = real_images.to(self.device)

                with torch.no_grad():
                    fake_images = self.sample(input_masks, num_samples=real_images.shape[0])

                self.visualize_predictions(input_masks, fake_images, real_images, mode)

            if mode == "train":
                lpips = self.train_lpips
            elif mode == "val":
                lpips = self.val_lpips
            else:
                lpips = self.test_lpips
            lpips_value = lpips.compute()
            lpips.reset()
            self.log(f"structural/lpips/{mode}", lpips_value)