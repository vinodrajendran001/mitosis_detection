import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

class MitosisDataset(Dataset):
    def __init__(self,img_list,mask_list,data_dir=None,type=None,transform=None):
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform
        self.type = type
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self,index):
        img_path = os.path.join(self.data_dir, self.type, 'input', self.img_list[index])
        mask_path = os.path.join(self.data_dir, self.type, 'output', self.mask_list[index])
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = np.array(img)
        mask = np.array(mask)
        mask[mask==255.0] = 1.0

        if self.transform:
            augmentation = self.transform(image=img, mask=mask)
            img = augmentation["image"]
            mask = augmentation["mask"]
            mask = torch.unsqueeze(mask,0)

            
        return img,mask