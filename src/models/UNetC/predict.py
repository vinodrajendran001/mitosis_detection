import os
import argparse
import torch
from torchvision import transforms
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
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



def parse_args():
    parser = argparse.ArgumentParser(
        description='Make segmentation predicitons'
    )
    parser.add_argument(
        '--model', type=str, default='UNet10.pt',
        help='model to use for inference'
    )
    parser.add_argument(
        '--input', type=str, help='input image file'
    )
    parser.add_argument(
        '--visualize', action='store_true', default=False,
        help='visualize the inference result'
    )
    args = parser.parse_args()
    return args

def predict(image, model):
    """Make prediction on image"""
    mean = 0.495
    std = 0.173
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(572, 572)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # transforms.Pad(30, padding_mode='reflect')
    ])
    im = image_transform(image)
    im = im.view(1, *im.shape)
    model.eval()
    y_pred = model(im)
    pred = torch.argmax(y_pred, dim=1)[0]
    return pred

def visualize(image, pred, label=None):
    """make visualization"""
    n_plot = 3
    fig = plt.figure()
    ax = fig.add_subplot(1, n_plot, 2)
    label[label == 255] = 1
    imgplot = plt.imshow(prediction_to_vis(label))
    ax.set_title('Actual Mask')
    ax = fig.add_subplot(1, n_plot, 3)
    print(np.unique(pred))
    print(pred.shape)
    # imgplot = plt.imshow(prediction_to_vis(pred))
    # ax.set_title('Predicted Mask')
    input_img = Image.fromarray(image).convert("RGBA")

    pred_mask = pred.squeeze(0).cpu().detach().numpy()
    # pred_mask = pred.transpose(1,0)
    predicted_mask = transform.resize(pred_mask, input_img.size, preserve_range=True, mode='constant')
    # predicted_mask = pred_mask.resize(input_img.size)
    imgplot = plt.imshow(prediction_to_vis(predicted_mask))
    ax.set_title('Predicted Mask')
    predicted_mask = predicted_mask.transpose(1,0)
    predicted_mask = prediction_to_vis(np.squeeze(predicted_mask))


    # predicted_mask = predicted_mask.resize(img.size)
    # print(predicted_mask.shape)
    # predicted_mask = predicted_mask.T
    # predicted_mask = prediction_to_vis(predicted_mask)
    
    predicted_mask = predicted_mask.convert("RGBA")
    print(input_img.size, predicted_mask.size)

    # overlay_img = Image.blend(input_img, predicted_mask, 0.9)
    ax = fig.add_subplot(1, n_plot, 1)
    imgplot = plt.imshow(image)
    ax.set_title('Image')
    fig.tight_layout()
    plt.savefig(f'reports/figures/{args.model[:-3]}_validation.png')
    plt.show()

if __name__ == '__main__':
    args = parse_args()

    # load images and labels
    path = args.input
    images = io.imread(path)
    label_path = path.replace("input","output")
    labels = io.imread(label_path)
    image = images
    label = labels

    # load model
    checkpoint_path = os.getcwd() + f'/models/{args.model}'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = UNet(2)
    model.load_state_dict(checkpoint['model_state_dict'])
    # make inference
    pred = predict(image, model)

    if args.visualize:
        # crop images for visualization
        dim = image.shape
        out_size = pred.shape[0]
        visualize(image, pred, label)

