import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def dice_loss(preds, targets):
    smooth = 1e-7
    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def iou_loss(preds, targets):
    smooth = 1e-7
    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou