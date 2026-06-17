import os
from angio_gen.utils.hospital_classifier import eval_hospital_classifier

datasets = [
    "aia_dataset",
    "aia_dataset_anon_pix2pix",
    "aia_dataset_anon_diffusion",
    "aia_dataset_anon_cn",
    "aia_dataset_anon_bc",
    "aia_dataset_anon_ln",
    "aia_dataset_anon_jpeg",
    "aia_dataset_anon_qu",
    "aia_dataset_anon_mf",
]

for ds in datasets:
    print("running for ", ds)
    eval_hospital_classifier(ds)

