import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import SegformerFeatureExtractor
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import argparse

from pathlib import Path
from segformer import SegformerFinetuner
from utils.data_loading import SemanticSegmentationDataset


def main(
        epochs: int = 5,
        batch_size: int = 1,
        data_dir: str = "./data",
        checkpoint_dir: str = "./checkpoints",
        accelerator: str = "gpu",
        devices: int = 1,
        min_delta: float = 0.0,
        patience: int = 10,
        metrics_interval: int = 10,
        val_check_interval: int = 5
):

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    feature_extractor.do_reduce_labels = False
    feature_extractor.size = 128

    train_dataset = SemanticSegmentationDataset(os.path.join(data_dir,'train'), feature_extractor)
    val_dataset = SemanticSegmentationDataset(os.path.join(data_dir,'val'), feature_extractor)


    num_workers = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    segformer_finetuner = SegformerFinetuner(
        train_dataset.id2label, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        test_dataloader=val_dataloader, 
        metrics_interval=metrics_interval,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=min_delta, 
        patience=patience, 
        verbose=False, 
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

    trainer = pl.Trainer(
        # gpus=1, 
        accelerator=accelerator, 
        devices=devices,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=epochs,
        # val_check_interval=len(train_dataloader),
        val_check_interval=val_check_interval,
        default_root_dir=checkpoint_dir
    )
    trainer.fit(segformer_finetuner)

    res = trainer.test(ckpt_path="best")


def get_args():
    parser = argparse.ArgumentParser(description='Train the Segformer images and target masks')
    parser.add_argument('--data_dir', '-d', metavar='D', type=str, help='Data directory')
    parser.add_argument('--checkpoint_dir', '-cp', metavar='CP', type=str, help='Checkpoint directory')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    # parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    # parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--accelerator', '-a', type=str, default="gpu", help='Enable gpu for training')
    parser.add_argument('--deviceid', '-id', type=int, default=1, help='GPU device id')
    parser.add_argument('--patience', '-p', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--metrics_interval', '-mi', type=int, default=10, help='Metrics interval')
    parser.add_argument('--val_metrics_interval', '-vmi', type=int, default=5, help='Validation metrics interval')


    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        accelerator=args.accelerator,
        devices=args.deviceid,
        patience= args.patience,
        metrics_interval=args.metrics_interval,
        val_check_interval=args.val_metrics_interval
        )

