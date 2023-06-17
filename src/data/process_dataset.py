# -*- coding: utf-8 -*-
import os
import shutil
import random
import math
import glob
import argparse

# Set the random seed
random_seed = 42
random.seed(random_seed)

def main(root_folder, train_folder, val_folder):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # list of all subfolders in the destination folder
    dest_subfolders = ['input', 'output', 'annotation', 'overlay']

    for subfolder in dest_subfolders:
        # Create train and val subfolders
        train_subfolder = os.path.join(train_folder, subfolder)
        val_subfolder = os.path.join(val_folder, subfolder)
        os.makedirs(train_subfolder, exist_ok=True)
        os.makedirs(val_subfolder, exist_ok=True)

    # Get a list of all subfolders in the root folder
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    for subfolder in subfolders:

        # Get a list of all png files in the subfolder
        png_files = [file for file in glob.glob(os.path.join(subfolder, '*.png')) if not file.endswith('_mask.png')]

        # Calculate the number of files for training and validation
        num_files = len(png_files)
        num_train = math.ceil(0.8 * num_files)
        num_val = num_files - num_train

        # Shuffle the list of png files
        random.shuffle(png_files)

        # Copy files to train subfolder
        train_files = png_files[:num_train]
        for file in train_files:
            filename = os.path.basename(file)
            shutil.copy2(file[:-4] + '.png', os.path.join(train_folder, 'input', filename))
            shutil.copy2(file[:-4] + '_mask.png', os.path.join(train_folder, 'output', filename))
            shutil.copy2(file[:-4] + '.jpg', os.path.join(train_folder, 'overlay', filename))
            shutil.copy2(file[:-4] + '.csv', os.path.join(train_folder, 'annotation', filename[:-4] + '.csv'))

        # Copy files to val subfolder
        val_files = png_files[num_train:]
        for file in val_files:
            filename = os.path.basename(file)
            shutil.copy2(file[:-4] + '.png', os.path.join(val_folder, 'input', filename))
            shutil.copy2(file[:-4] + '_mask.png', os.path.join(val_folder, 'output', filename))
            shutil.copy2(file[:-4] + '.jpg', os.path.join(val_folder, 'overlay', filename))
            shutil.copy2(file[:-4] + '.csv', os.path.join(val_folder, 'annotation', filename[:-4] + '.csv'))

if __name__ == '__main__':

    root_folder = '/data1/vinod/mitosis/data/raw'
    train_folder = '/data1/vinod/mitosis/data/processed/train'
    val_folder = '/data1/vinod/mitosis/data/processed/val'

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Copy files to train and val folders')
    parser.add_argument('--root', '-r', help='Root folder path')
    parser.add_argument('--train', '-t', help='Train folder path')
    parser.add_argument('--val', '-v', help='Val folder path')
    args = parser.parse_args()

    # Read the arguments
    root_folder = args.root
    train_folder = args.train
    val_folder = args.val

    # Create train and val folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    main(root_folder, train_folder, val_folder)
