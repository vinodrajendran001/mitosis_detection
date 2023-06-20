import torch
from torch import nn
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor

        self.classes_csv_file = os.path.join(self.root_dir, "_classes.csv")
        with open(self.classes_csv_file, 'r') as fid:
            data = [l.split(',') for i,l in enumerate(fid) if i !=0]
        self.id2label = {x[0]:x[1] for x in data}
        
        image_file_names = [f for f in os.listdir(os.path.join(self.root_dir, 'input')) if '.png' in f]
        mask_file_names = [f for f in os.listdir(os.path.join(self.root_dir, 'output')) if '.png' in f]
        
        self.images = sorted(image_file_names)
        self.masks = sorted(mask_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = Image.open(os.path.join(self.root_dir, 'input', self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.root_dir, 'output', self.masks[idx]))

        # Convert the image to a NumPy array
        segmentation_array = np.array(segmentation_map)
        # Replace all the 255 values with 1
        segmentation_array[segmentation_array == 255] = 1
        # Convert the modified array back to an image
        segmentation_map = Image.fromarray(segmentation_array)
        
        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs