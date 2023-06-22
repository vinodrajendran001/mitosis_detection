import argparse
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from PIL import Image
from unet import UNet

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
    model,
    img_path: str = "./data",
    save_dir: str = "./save"
):

    img = Image.open(img_path)

    test_transform = A.Compose([A.Resize(512,512),
                           A.Normalize(mean=(0,0,0),std=(1,1,1),max_pixel_value=255),
                           ToTensorV2()])
    
    test_image = test_transform(image = np.asarray(img))
    test_image = test_image["image"].unsqueeze(0)
    test_image = test_image.to(device)

    
    mask_path = img_path.replace("input","output")
    mask = Image.open(mask_path)
    test_mask = np.copy(np.asarray(mask))
    test_mask[test_mask==255] = 1

    pred_mask = model(test_image)
    pred_mask = pred_mask.squeeze(0).cpu().detach().numpy()
    pred_mask = pred_mask.transpose(1,2,0)
    pred_mask[pred_mask < 0]=0
    pred_mask[pred_mask > 0]=1

    predicted_mask = prediction_to_vis(np.squeeze(pred_mask))
    predicted_mask = predicted_mask.resize(img.size)
    input_img = img.convert("RGBA")
    predicted_mask = predicted_mask.convert("RGBA")
    overlay_img = Image.blend(input_img, predicted_mask, 0.5)

    fig, axarr = plt.subplots(1, 3, figsize=(10, 4)) 
    axarr[0].imshow(prediction_to_vis(test_mask))
    axarr[0].set_title('Actual Mask')
    axarr[0].axis('off')
    axarr[1].imshow(prediction_to_vis(np.squeeze(pred_mask)))
    axarr[1].set_title('Predicted Mask')
    axarr[1].axis('off')
    axarr[2].imshow(overlay_img)
    axarr[2].set_title('Overlay Image')
    axarr[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'unet_output.png'))

def get_args():
    parser = argparse.ArgumentParser(description='Predict the UNet on images and target masks')
    parser.add_argument('--image', '-d', metavar='D', type=str, help='Data directory')
    parser.add_argument('--model_load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--save_dir', '-s', type=str, default=False, help='Save result image')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(in_chnls = 3, n_classes = 1)
    model.load_state_dict(torch.load(args.model_load))
    model.to(device=device)
    model.eval()

    main(
        model,
        img_path=args.image,
        save_dir=args.save_dir
        )