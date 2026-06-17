import argparse
import torch
import os
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from angio_gen.data.dataset import ICABinaryDatasetTest
from angio_gen.data.transforms import apply_test_transform
from angio_gen.utils.writer import ImagePredictionWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--resume", type=str, help="Path to model checkpoint")
    parser.add_argument("--dataset_dir", type=str, default="re_dataset", help="Path to test dataset")
    parser.add_argument("--num_workers", type=int, default=31, help="Number of workers")

    return parser.parse_args()


def test(Model, output_dir):
    args = parse_args()

    test_dataset = ICABinaryDatasetTest(root=args.dataset_dir, transform=apply_test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    os.makedirs(output_dir, exist_ok=True)
    pred_writer = ImagePredictionWriter(output_dir=output_dir, write_interval="batch")
    
    torch.set_float32_matmul_precision('medium')
    model = Model.load_from_checkpoint(args.resume)
    model.eval()
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
        callbacks=[pred_writer],
        enable_progress_bar=True
    )
    trainer.predict(model, test_dataloader, return_predictions=False)


def test_hospital_clf(model, dm):
    torch.set_float32_matmul_precision('medium')
    model.eval()
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True
    )
    trainer.test(model, datamodule=dm)
