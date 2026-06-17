import torch
import os

from angio_gen.data.hospital_dataset import HospitalDataModule
from angio_gen.utils.metrics import compute_classwise_energy_distance

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


for ds in datasets:
    dm = HospitalDataModule(dataset_dir=ds, metadata_path=os.path.join(ds, "metadata.json"), batch_size=128)
    print(f"{ds} {compute_classwise_energy_distance(dm)}")
