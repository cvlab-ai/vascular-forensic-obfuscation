import torch
import torch.nn as nn
import torchvision.utils as utils
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2, ndf=64):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final_layer = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1) 

    def forward(self, input_mask, image):
        x = torch.cat([input_mask, image], dim=1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output = self.final_layer(x)

        return output

def prepare_for_lpips(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    
    return torch.clamp(x, -1.0, 1.0)

class CGANModule(pl.LightningModule):
    def __init__(self, lr=2e-4, b1=0.5, b2=0.999, lambda_l1=100.0, use_scheduler=True, image_size=512):
        super().__init__()
        self.save_hyperparameters()

        self.generator = smp.Unet(
            encoder_name='resnet18',
            encoder_weights='imagenet',
            in_channels=1,
            classes=1,
            activation=None
        )

        self.discriminator = PatchDiscriminator(in_channels=1 + 1, ndf=64)
        self.criterion_GAN = nn.BCEWithLogitsLoss() 
        self.criterion_L1 = nn.L1Loss()

        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.lambda_l1 = lambda_l1
        self.use_scheduler = use_scheduler
        self.image_size = image_size

        self.automatic_optimization = False
        self._last_viz_batch = None

        self.train_lpips = LPIPS()
        self.val_lpips = LPIPS()
        self.test_lpips = LPIPS()

        self.fid = FID()

    def forward(self, x):
        return self.generator(x)
    
    def inference(self, x):
        for m in self.generator.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        real_image, input_mask = batch

        self.visualize_gt(batch, batch_idx)

        fake_image = torch.tanh(self.generator(input_mask))
        self.visualize_predictions(input_mask, fake_image, real_image, batch_idx, "train")
        
        if self._last_viz_batch is None or (batch_idx % self.trainer.log_every_n_steps == 0):
            self._last_viz_batch = (real_image[:4].detach().cpu(), input_mask[:4].detach().cpu())

        opt_g, opt_d = self.optimizers()

        # discriminator
        opt_d.zero_grad()
        pred_real = self.discriminator(input_mask, real_image)
        valid_targets = torch.ones_like(pred_real, requires_grad=False)
        d_loss_real = self.criterion_GAN(pred_real, valid_targets)
        pred_fake = self.discriminator(input_mask, fake_image.detach())
        fake_targets = torch.zeros_like(pred_fake, requires_grad=False)
        d_loss_fake = self.criterion_GAN(pred_fake, fake_targets)
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        self.manual_backward(d_loss)
        opt_d.step()

        # Generator
        opt_g.zero_grad()
        pred_fake_for_g = self.discriminator(input_mask, fake_image)
        g_loss_gan = self.criterion_GAN(pred_fake_for_g, valid_targets)
        g_loss_l1 = self.criterion_L1(fake_image, real_image)
        g_loss = g_loss_gan + self.lambda_l1 * g_loss_l1

        self.manual_backward(g_loss)
        opt_g.step()

        self.log_dict({
            'train/g_loss': g_loss,
            'train/d_loss': d_loss,
            'train/g_loss_gan': g_loss_gan,
            'structural/train_g_loss_l1': g_loss_l1,
            'train/d_loss_real': d_loss_real,
            'train/d_loss_fake': d_loss_fake,
        }, prog_bar=True, on_step=True, on_epoch=True)

        self.train_lpips.update(prepare_for_lpips(fake_image), prepare_for_lpips(real_image))

        return d_loss
    
    def validation_step(self, batch, batch_idx):
        real_image, input_mask = batch

        fake_image = torch.tanh(self.generator(input_mask))
        self.visualize_predictions(input_mask, fake_image, real_image, batch_idx, "val")

        opt_g, opt_d = self.optimizers()
        opt_d.zero_grad()
        opt_g.zero_grad()

        pred_real = self.discriminator(input_mask, real_image)
        valid_targets = torch.ones_like(pred_real, requires_grad=False)
        d_loss_real = self.criterion_GAN(pred_real, valid_targets)
        pred_fake = self.discriminator(input_mask, fake_image.detach())
        fake_targets = torch.zeros_like(pred_fake, requires_grad=False)
        d_loss_fake = self.criterion_GAN(pred_fake, fake_targets)
        d_loss = (d_loss_real + d_loss_fake) * 0.5

        g_loss_gan = self.criterion_GAN(pred_fake, valid_targets)
        g_loss_l1 = self.criterion_L1(fake_image, real_image)
        g_loss = g_loss_gan + self.lambda_l1 * g_loss_l1

        self.log_dict({
            'val/g_loss': g_loss,
            'val/d_loss': d_loss,
            'val/g_loss_gan': g_loss_gan,
            'structural/val_g_loss_l1': g_loss_l1,
            'val/d_loss_real': d_loss_real,
            'val/d_loss_fake': d_loss_fake,
        }, prog_bar=True, on_step=True, on_epoch=True)

        self.val_lpips.update(prepare_for_lpips(fake_image), prepare_for_lpips(real_image))

        return d_loss
    
    def test_step(self, batch, batch_idx):
        real_image, input_mask = batch
        fake_image = torch.tanh(self.generator(input_mask))
        self.test_lpips.update(prepare_for_lpips(fake_image), prepare_for_lpips(real_image))

        fake_image = ((fake_image + 1) / 2 * 255).to(torch.uint8).repeat(1, 3, 1, 1)
        real_image = ((real_image + 1) / 2 * 255).to(torch.uint8).repeat(1, 3, 1, 1)

        self.fid.update(fake_image, real=False)
        self.fid.update(real_image, real=True)


    def on_train_epoch_end(self):
        lpips_value = self.train_lpips.compute()
        self.train_lpips.reset()
        self.log("structural/train_lpips", lpips_value)

    def on_validation_epoch_end(self):
        lpips_value = self.val_lpips.compute()
        self.val_lpips.reset()
        self.log("structural/val_lpips", lpips_value)

    def on_test_epoch_end(self):
        fid_value = self.fid.compute()
        lpips_value = self.test_lpips.compute()

        self.log("test/fid", fid_value, on_epoch=True)
        self.log("test/lpips", lpips_value, on_epoch=True)
    
    def predict_step(self, batch, batch_idx):
        input_mask, filename = batch
        fake_image = torch.tanh(self.generator(input_mask))
        fake_image = (fake_image + 1) / 2
        return {"image_tensor": fake_image, "filename": filename}

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        optimizers = [opt_g, opt_d]
 
        if self.use_scheduler:
            scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=50, gamma=0.5)
            scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=50, gamma=0.5)
            
            schedulers = [
                {'scheduler': scheduler_g, 'interval': 'epoch'},
                {'scheduler': scheduler_d, 'interval': 'epoch'}
            ]
            return optimizers, schedulers
        return optimizers, []

    def visualize_gt(self, batch, batch_idx):
        images, masks = batch

        if batch_idx == 1:
            pil_images = [to_pil_image(torch.clamp((image + 1.0) / 2, 0, 1)) for image in images]
            pil_masks = [to_pil_image(torch.clamp((mask + 1.0) / 2, 0, 1)) for mask in masks]

            grid = utils.make_grid([pil_to_tensor(image) for image in (pil_images + pil_masks)])
            self.logger.experiment.add_image("gt/images", grid, self.current_epoch)

    def visualize_predictions(self, input_masks, fake_images, real_images, batch_idx, mode):
        
        if batch_idx == 1:
            pil_real_images = [to_pil_image(torch.clamp((image + 1.0) / 2, 0, 1)) for image in real_images]
            pil_input_masks = [to_pil_image(torch.clamp((mask + 1.0) / 2, 0, 1)) for mask in input_masks]
            pil_fake_images = [to_pil_image(torch.clamp((image + 1.0) / 2, 0, 1)) for image in fake_images]

            grid = utils.make_grid([pil_to_tensor(image) for image in (pil_input_masks + pil_real_images + pil_fake_images)])
            self.logger.experiment.add_image(f"{mode}_predictions/images", grid, self.current_epoch)