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
        data_dir: str = "./data",
        checkpoint_dir: str = "./checkpoints",
        batch_size: int = 1,
        metrics_interval: int = 10
):
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    feature_extractor.do_reduce_labels = False
    feature_extractor.size = 128

    val_dataset = SemanticSegmentationDataset(os.path.join(data_dir,'val'), feature_extractor)

    num_workers = os.cpu_count()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    segformer_finetuner = SegformerFinetuner(
        val_dataset.id2label,
        val_dataloader=val_dataloader, 
        test_dataloader=val_dataloader, 
        metrics_interval=metrics_interval,
    )

    model = segformer_finetuner.load_from_checkpoint(checkpoint_dir)

    # disable randomness, dropout, etc...
    # model.eval()

    # predict with the model
    model.test(ckpt_path="best")


def get_args():
    parser = argparse.ArgumentParser(description='Predict the Segformer images and target masks')
    parser.add_argument('--data_dir', '-d', metavar='D', type=str, help='Data directory')
    parser.add_argument('--checkpoint_dir', '-cp', metavar='CP', type=str, help='Checkpoint directory')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        )