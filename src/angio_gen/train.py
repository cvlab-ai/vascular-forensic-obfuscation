import os
import torch
import argparse
import random
from clearml import Task
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from angio_gen.data.dataset import ICABinaryDataset
from angio_gen.data.transforms import apply_train_transforms, apply_val_transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--dataset_path', type=str, default="re_dataset", help='Dataset path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=31, help='Number of workers')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Dataset split ratio')
    parser.add_argument('--save_every_n_epochs', type=int, default=20, help='Saving frequency')
    parser.add_argument('--ckpt_dir', type=str, default="", help='Relative directory path for saving checkpoints.')
    parser.add_argument('--no_split', action="store_true", help='If not specified, split dataset into training and validation.')

    return parser.parse_args()


def train_with_clearml(task_name, *args, **kwargs):
    Task.init(task_name=task_name, project_name="data-generation-from-binary")
    if 'datamodule' in kwargs: return train_with_custom_dataset(task_name, *args, **kwargs)
    else: return train(task_name, *args)


def train_with_custom_dataset(task_name, Model, datamodule, gradient_clip=True, early_stopping=True):
    args = parse_args()

    early_stop_callback = EarlyStopping(
        monitor="loss/val_loss",
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode="min"
    )

    ckpt_dir = args.ckpt_dir if args.ckpt_dir != "" else f"checkpoints/{task_name}"
    torch.set_float32_matmul_precision('medium')
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
    )
    callbacks = [checkpoint_callback]
    if early_stop_callback: callbacks.append(early_stop_callback)
    model = Model()
    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
        gradient_clip_val=0.5 if gradient_clip else None
    )
    trainer.fit(model, datamodule=datamodule)
    return model


def train(task_name, Model):
    args = parse_args()

    if args.no_split:
        dataset = ICABinaryDataset(root=args.dataset_path, transforms=apply_train_transforms)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    else:
        train_dataset = ICABinaryDataset(root=args.dataset_path, _set="training", transforms=apply_train_transforms)
        val_dataset = ICABinaryDataset(root=args.dataset_path, _set="validation", transforms=apply_val_transforms)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir != "" else f"checkpoints/{task_name}"
    torch.set_float32_matmul_precision('medium')
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
    )
    model = Model()
    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true"
    )

    if args.no_split:
        trainer.fit(model, dataloader)
    else:
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, val_dataloader)
    return model