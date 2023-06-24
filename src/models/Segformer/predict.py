import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import SegformerFeatureExtractor
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from pathlib import Path
from segformer import SegformerFinetuner
from utils.data_loading import SemanticSegmentationDataset
from PIL import Image

import matplotlib.pyplot as plt

color_map = {
    0:(0,0,0),
    1:(255,0,0),
}

def prediction_to_vis(prediction):
    vis_shape = prediction.shape + (3,)
    vis = np.zeros(vis_shape)
    for i,c in color_map.items():
        vis[prediction == i] = color_map[i]
    return Image.fromarray(vis.astype(np.uint8))

def main(
        data_dir: str = "./data",
        checkpoint_dir: str = "./checkpoints",
        test_idx: int = 1,
        save_dir: str = "./output",
        metrics_interval: int = 10
):
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    feature_extractor.do_reduce_labels = False
    feature_extractor.size = 128

    val_dataset = SemanticSegmentationDataset(os.path.join(data_dir,'val'), feature_extractor)

    num_workers = os.cpu_count()
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)


    segformer_finetuner = SegformerFinetuner(
        val_dataset.id2label, 
        metrics_interval=metrics_interval,
    )

    segformer_finetuner.load_from_checkpoint(checkpoint_dir)


    #Predict on a val image and overlay the mask on the original image
    test_idx = test_idx

    input_image_file = os.path.join('/data1/vinod/mitosis/data/processed','val', 'input', val_dataset.images[test_idx])
    input_image = Image.open(input_image_file)
    test_batch = val_dataset[test_idx]
    images, masks = test_batch['pixel_values'], test_batch['labels']
    images = torch.unsqueeze(images, 0)
    masks = torch.unsqueeze(masks, 0)
    outputs = segformer_finetuner.model(images, masks)
        
    loss, logits = outputs[0], outputs[1]

    upsampled_logits = nn.functional.interpolate(
        logits, 
        size=masks.shape[-2:], 
        mode="bilinear", 
        align_corners=False
    )
    predicted_mask = upsampled_logits.argmax(dim=1).cpu().numpy()
    mask = prediction_to_vis(np.squeeze(predicted_mask))
    mask = mask.resize(input_image.size)
    mask = mask.convert("RGBA")
    inp_image = input_image.convert("RGBA")
    overlay_img = Image.blend(inp_image, mask, 0.5)
    fig, axarr = plt.subplots(1, 3, figsize=(10, 4))
    
    axarr[1].imshow(prediction_to_vis(masks[0,:,:]))
    axarr[1].set_title('Actual Mask')
    axarr[1].axis('off')
    axarr[2].imshow(prediction_to_vis(predicted_mask[0,:,:]))
    axarr[2].set_title('Predicted Mask')
    axarr[2].axis('off')
    axarr[0].imshow(input_image)
    axarr[0].set_title('Image')
    axarr[0].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'segformer_output.png'))


def get_args():
    parser = argparse.ArgumentParser(description='Predict the Segformer images and target masks')
    parser.add_argument('--data_dir', '-d', metavar='D', type=str, help='Data directory')
    parser.add_argument('--checkpoint_dir', '-cp', metavar='CP', type=str, help='Checkpoint directory')
    parser.add_argument('--test_idx', '-id', type=int, default=1, help='Image ID to predict')
    parser.add_argument('--save_dir', '-s', type=str, help='Save the output')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        test_idx=args.test_idx,
        save_dir=args.save_dir
        )