from angio_gen.train import train_with_clearml
from angio_gen.models.cgan import CGANModule


train_with_clearml(
    "cgan",
    CGANModule,
)