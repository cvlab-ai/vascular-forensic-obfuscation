import os
from pytorch_lightning.callbacks import BasePredictionWriter
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

class ImagePredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.postprocess = transforms.Resize((512, 512))

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        image_tensors = prediction["image_tensor"]
        filenames = prediction["filename"]

        for i, filename in enumerate(filenames):
            image_tensor = image_tensors[i, ...]
            processed_img = self.postprocess(image_tensor)
            image = to_pil_image(processed_img)
            save_path = os.path.join(self.output_dir, filename)            
            image.save(save_path)
