import argparse
import os
import logging
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import torch.nn as nn
import albumentations as A
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from albumentations.pytorch import ToTensorV2 
from torch.utils.data import DataLoader, Dataset


from pathlib import Path
from unet import UNet
from utils.data_loading import MitosisDataset



def train_model(model,dataloader,batch_size,criterion,optimizer,gradient_clipping):
    model.train()
    train_running_loss = 0.0
    for j,img_mask in enumerate(tqdm(dataloader)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)       
        y_pred = model(img)
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        loss = criterion(y_pred,mask)
        train_running_loss += loss.item() * batch_size
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss / (j+1)
    return train_loss

def val_model(model,dataloader,batch_size,criterion):
    model.eval()
    val_running_loss = 0
    with torch.no_grad():
        for j,img_mask in enumerate(tqdm(dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            y_pred = model(img)
            
            loss = criterion(y_pred,mask)
            val_running_loss += loss.item() * batch_size

            
        val_loss = val_running_loss / (j+1)

    return val_loss

def main(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        img_size: int = 512,
        data_dir: str = "./data",
        checkpoint_dir: str = "./checkpoints",
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0
):
    # 1. Create dataset

    train_img_lst = sorted(os.listdir(os.path.join(data_dir, 'train', 'input'))) # "./train"
    train_mask_lst = sorted(os.listdir(os.path.join(data_dir, 'train', 'output'))) # "./train_masks"
    val_img_lst = sorted(os.listdir(os.path.join(data_dir, 'val', 'input'))) # "./val"
    val_mask_lst = sorted(os.listdir(os.path.join(data_dir, 'val', 'output'))) # "./val_masks"


    train_transform = A.Compose([A.Resize(int(img_size),int(img_size)), 
                                A.Rotate(limit=15,p=0.1),
                                A.HorizontalFlip(p=0.5),
                                A.Normalize(mean=(0,0,0),std=(1,1,1),max_pixel_value=255),
                                ToTensorV2()])

    val_transform = A.Compose([A.Resize(int(img_size),int(img_size)),
                            A.Normalize(mean=(0,0,0),std=(1,1,1),max_pixel_value=255),
                            ToTensorV2()])


    train_dataset = MitosisDataset(train_img_lst, train_mask_lst, data_dir=data_dir, type='train', transform = train_transform)
    val_dataset = MitosisDataset(val_img_lst, val_mask_lst, data_dir=data_dir, type='val', transform = val_transform)

    # 2. count of train / validation sets
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    logging.info(f'''Starting training:
        Training datset:         {n_train}
        Validation dataset:      {n_val}
    ''')

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_dataloader = DataLoader(train_dataset,shuffle=True,**loader_args)
    val_dataloader = DataLoader(val_dataset,shuffle=False,**loader_args)

    # (Initialize logging)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Image resized:   {img_size}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 
    criterion = nn.BCEWithLogitsLoss()
    train_loss_lst = []
    val_loss_lst = []
     

    # 5. Begin training
    for i in tqdm(range(epochs)):
        train_loss = train_model(model=model,dataloader=train_dataloader,batch_size=batch_size,criterion=criterion,optimizer=optimizer,gradient_clipping=gradient_clipping)
        val_loss = val_model(model=model,dataloader=val_dataloader,batch_size=batch_size,criterion=criterion)
        train_loss_lst.append(train_loss)
        val_loss_lst.append(val_loss)
        print(f" Train Loss : {train_loss:.4f}")
        print(f" Validation Loss : {val_loss:.4f}")

        if save_checkpoint and i%10 == 0:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(checkpoint_dir + '/' + 'checkpoint_epoch{}.pth'.format(i)))
            logging.info(f'Checkpoint {i} saved!')

    plt.plot(train_loss_lst, color="green", label='train loss')
    plt.plot(val_loss_lst, color="red", label='validation loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("reports/figures/unet_training.png")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_dir', '-d', metavar='D', type=str, help='Data directory')
    parser.add_argument('--checkpoint_dir', '-cp', metavar='CP', type=str, help='Checkpoint directory')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--image_size', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()




if __name__ == '__main__':
    args = get_args()
    # dir_data = Path('/data1/vinod/mitosis/data/processed')
    # dir_checkpoint = Path('models/UNet/version1.0')
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(in_chnls = 3, n_classes = args.classes).to(device)
    model = model.to(memory_format=torch.channels_last)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    main(
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_size=args.image_size,
                data_dir=args.data_dir,
                checkpoint_dir=args.checkpoint_dir
            )

