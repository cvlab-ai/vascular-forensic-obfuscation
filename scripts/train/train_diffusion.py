from angio_gen.train import train_with_clearml
from angio_gen.models.diffusion import DiffusionModule


train_with_clearml(
    "diffusion",
    DiffusionModule,
)