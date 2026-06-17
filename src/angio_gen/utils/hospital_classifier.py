import os
from angio_gen.train import train_with_clearml
from angio_gen.test import test_hospital_clf
from angio_gen.models.hospital_classifier import HospitalClassifier
from angio_gen.data.hospital_dataset import HospitalDataModule
from clearml import Task

def eval_hospital_classifier(root):
    dm = HospitalDataModule(dataset_dir=root, metadata_path=os.path.join(root, "metadata.json"), batch_size=128)
    model, task = train_with_clearml(
        f"AIA ({root})",
        HospitalClassifier,
        datamodule=dm
    )
    test_hospital_clf(
        model,
        dm
    )
    task.close()
