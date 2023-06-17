# import libraries
import os
import sys
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_binary_mask(image_size, coordinates):
    mask = np.zeros(image_size, dtype=np.uint8)
    for coord in coordinates:
        for i in range(0, len(coord), 2):
            x, y = coord[i], coord[i+1]
            mask[x, y] = 255
    return mask

if __name__ == '__main__':

    # Path to the data folder
    data_folder = sys.argv[1]

    # Iterate over images in the folder
    for sub_folder in os.listdir(data_folder):
        for filename in os.listdir(os.path.join(data_folder,sub_folder)):
            if filename.endswith(".png"):
                image_path = os.path.join(data_folder, sub_folder, filename)

                # Load the image
                image = Image.open(image_path)
                image_size = image.size

                # Get the corresponding CSV file
                csv_filename = os.path.splitext(filename)[0] + ".csv"
                csv_path = os.path.join(data_folder, sub_folder, csv_filename)
                # Check if the CSV file exists
                if os.path.exists(csv_path):
                    # Read the CSV file and extract the coordinates
                    coordinates = []
                    with open(csv_path, 'r') as csv_file:
                        csv_reader = csv.reader(csv_file)
                        for row in csv_reader:
                            coordinates.append([int(coord) for coord in row])

                    # Create the binary mask
                    mask = create_binary_mask(image_size, coordinates)

                    # Transpose the mask array
                    mask = np.transpose(mask)

                    # Save the mask as a binary image
                    mask_image = Image.fromarray(mask, mode='L')
                    mask_filename = os.path.splitext(filename)[0] + "_mask.png"
                    mask_path = os.path.join(data_folder, sub_folder, mask_filename)
                    mask_image.save(mask_path)
                    print(f"Binary mask created for {filename}.")
                else:
                    print(f"CSV file not found for {filename}. Skipping.")

    print("Binary mask created successfully!")