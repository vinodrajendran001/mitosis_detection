import os
from tqdm import tqdm

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from skimage.segmentation import find_boundaries
import numpy as np
import matplotlib.pyplot as plt


from utils.augmentation import (
    DoubleCompose, DoubleToTensor,
    DoubleHorizontalFlip, DoubleVerticalFlip, DoubleElasticTransform
)


class MitosisDataset(Dataset):

    def __init__(
        self, root_dir=None,
        image_mask_transform=None, image_transform=None, mask_transform=None,
        data_type='train', in_size=512, out_size=512,
        w0=10, sigma=5, weight_map_dir=None
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            image_mask_transform (callable, optional): Optional
            transform to be applied on images and mask label simultaneuosly.
            image_transform (callable, optional): Optional
            transform to be applied on images.
            mask_transform (callable, optional): Optional
            transform to be applied on mask labels.
            data_type (string): either 'train' or 'test'
            in_size (int): input size of image
            out_size (int): output size of segmentation map
        """
        self.root_dir = os.getcwd() if not root_dir else root_dir
        path =self.root_dir
        self.train_path = os.path.join(path, 'train', 'input')
        self.t_mask_path = os.path.join(path, 'train', 'output')
        self.val_path = os.path.join(path, 'val', 'input')
        self.v_mask_path = os.path.join(path, 'val', 'output')

        self.data_type = data_type
        self.image_mask_transform = image_mask_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.weight_transform = self.mask_transform
        if self.data_type == 'validate':
            self.weight_transform = transforms.Compose(
                self.mask_transform.transforms[1:]
            )
        self.n_classes = 2
        # self.images = io.imread(self.train_path)
        # self.masks = io.imread(self.mask_path)

        # n = int(np.ceil(self.images.shape[0] * pct))

        if self.data_type == 'train':
            self.images = [io.imread(os.path.join(self.train_path, f)) for f in os.listdir(self.train_path) if '.png' in f]
            self.masks = [io.imread(os.path.join(self.t_mask_path, f)) for f in os.listdir(self.t_mask_path) if '.png' in f]

        elif self.data_type == 'validate':
            self.images = [io.imread(os.path.join(self.val_path, f)) for f in os.listdir(self.val_path) if '.png' in f]
            self.masks = [io.imread(os.path.join(self.v_mask_path, f)) for f in os.listdir(self.v_mask_path) if '.png' in f]

        resized_img_arrays = []
        for image_array in self.images:
            resized_array = transform.resize(image_array, (in_size, out_size), preserve_range=True)
            resized_img_arrays.append(resized_array)

        resized_mask_arrays = []
        for mask_array in self.masks:
            resized_array = transform.resize(mask_array, (in_size, out_size), preserve_range=True)
            resized_mask_arrays.append(resized_array)

        self.mean = np.average(resized_img_arrays)
        self.std = np.std(resized_img_arrays)
        self.w0 = w0
        self.sigma = sigma
        # if weight_map_dir:
        #     self.weight_map = torch.load(weight_map_dir)
        #     print(self.weight_map)
        # if not weight_map_dir:
        self.images = np.stack(resized_img_arrays)
        self.masks = np.stack(resized_mask_arrays)
        self.weight_map = self._get_weights(self.w0, self.sigma)
        # torch.save(self.weight_map, 'weight_map.pt')

        self.in_size = in_size
        self.out_size = out_size
        # print(self.images.shape, 'images')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        """Returns a image sample from the dataset
        """
        image = self.images[idx]
        mask = self.masks[idx]
        weight = self.weight_map[idx]

        if self.image_mask_transform:
            image, mask, weight = self.image_mask_transform(
                image, mask, weight
            )
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            weight = self.weight_transform(mask)

        # img = Image.fromarray(255*label[0].numpy())
        # img.show()
        sample = {'image': image, 'mask': mask, 'weight': weight}

        return sample

    def _get_weights(self, w0, sigma):
        class_weight = self._get_class_weight(self.masks)
        # boundary_weight = self._get_boundary_weight(self.masks, w0, sigma)
        return class_weight  # + boundary_weight

    def _get_class_weight(self, target):
        n, H, W = target.shape
        weight = torch.zeros(n, H, W)
        for i in range(self.n_classes):
            i_t = i * torch.ones([n, H, W], dtype=torch.long)
            loc_i = (torch.Tensor(target // 255) == i_t).to(torch.long)
            count_i = loc_i.view(n, -1).sum(1)
            total = H * W
            weight_i = total / count_i
            weight_t = weight_i.view(-1, 1, 1) * loc_i
            weight += weight_t
        return weight

    def _get_boundary_weight(self, target, w0=10, sigma=5):
        """This implementation is very computationally intensive!
        about 30 minutes per 512x512 image
        """
        print('Calculating boundary weight...')
        n, H, W = target.shape
        weight = torch.zeros(n, H, W)
        ix, iy = np.meshgrid(np.arange(H), np.arange(W))
        ix, iy = np.c_[ix.ravel(), iy.ravel()].T
        for i, t in enumerate(tqdm(target)):
            boundary = find_boundaries(t, mode='inner')
            bound_x, bound_y = np.where(boundary is True)
            # broadcast boundary x pixel
            dx = (ix.reshape(1, -1) - bound_x.reshape(-1, 1)) ** 2
            dy = (iy.reshape(1, -1) - bound_y.reshape(-1, 1)) ** 2
            d = dx + dy
            # distance to 2 closest cells
            d2 = np.sqrt(np.partition(d, 2, axis=0)[:2, ])
            dsum = d2.sum(0).reshape(H, W)
            weight[i] = torch.Tensor(w0 * np.exp(-dsum**2 / (2 * sigma**2)))
        return weight
