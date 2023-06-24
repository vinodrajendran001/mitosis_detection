Mitosis_Detection
==============================

Develop an AI algorithm that support histophathologists in the detection of Mitosis

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── Segformer
    │   │   └── UNet
    │   │   └── UNetC
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

General domain description
==========================

Mitotic count is an important parameter in breast cancer grading as it gives an evaluation of the agressiveness of the 
tumour. However, mitosis detection is a challenging problem and has not been addressed well in the literature. This is due
to the fact that mitosis are small objects with a large variety of shape configurations. The four main phases of a mitosis
are prophase, metaphase, anaphase and telophase. The shape of the nucleus  is very different depending on the phase of the
mitosis. On its last stage, the telophase, a mitosis has two distinct nuclei, but they are not yet full individual cells. 
A mitosis in telophase must be counted as a single mitosis, it should not be miscounted as two mitosis. Artefacts are also
common and should not be confused with mitosis. 

The data you receive are H&E stained histological images of different breast cancers prepared on 5 
slides and each slide is scanned on two scanners. The data of scanner A can be found in the folders A*; 
data of scanner H can be found in folders H* (e.g. data of slide 0 scanned on scanner A is in found A00_v2,
the same slide scanned on scanner H is found in H00_v2).

Our objective is to develop an algorithm that support histophathologists in the detecting of
mitosis on these type of images.

Data description
================

The data is organized as follows.

Each scanned image consists of a raw image, the desired output
as a pixel list and an overlay of both.

For example:

Input image: A00_00.png
Annotations: A00_00.csv
Overlay: A00_00.jpg

The annotations are organized one mitosis per line.
Each line lists the coordinates of all pixels belonging
to one mitosis segment.

```
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
```

## Create virtual environment

```
user@systemname:project_directory$ conda craete -n seg python=3.9

(seg) user@systemname:project_directory$ conda activate seg

(seg) user@systemname:project_directory$ conda install nb_conda_kernels

(seg) user@systemname:project_directory$ pip install -r requirements.txt
```

## Solution

### Data stats

![Scanners Stats](reports/figures/data_count.png)


### Create binary mask

```
(seg) user@systemname:project_directory$ python src/data/create_binarymask.py --root /path/to/root/folder
```

### Process dataset for train and val 

```
(seg) user@systemname:project_directory$ python src/data/process_dataset.py --root /path/to/root/folder --train /path/to/train/folder --val /path/to/val/folder
```

### UNet model (Regressor)

```
(seg) user@systemname:project_directory$ python src/models/UNet/train.py -d /path/to/root/folder -cp /path/to/save/checkpoints -e 100 -b 8 -l 3e-4 -s 572 -c 1
```

![Model performance](reports/figures/unet_training.png)

| Test Metric         | Score                   |
| :---:               | :---:                   | 
| train_loss          | 0.00198                 |
| valid_loss          | 0.0334                  |

#### Prediction

```
(seg) user@systemname:project_directory$ python src/models/UNet/predict.py --image /path/to/input/image.png --model_load /path/to/checkpoint/file.pth --save_dir /path/to/save/output
```

![UNet output](reports/figures/unet_output.png)

### UNet model (Classification)

#### Training
```
python src/models/UNetC/train.py --batch-size 8 --test-batch-size 8 --epochs 10 --lr 0.0001 --n-classes 2 --data_dir /path/to/data/folder
```

#### Prediction
```
python src/models/UNetC/predict.py --model UNet100.pt --input /path/to/image/file.png --visualize
```

| Test Metric         | Epoch 3 | Epoch 5 | Epoch 8 | Epoch 10 |
| :---:               | :---:   | :---:   | :---:   | :---:    | 
| test_loss           | 0.0523  | 0.0063  | 0.0122  | 0.0058   |
| test_mean_iou       | 0.50    | 0.0790  | 0.9500  | 0.9500   |
| test_mean_accuracy  | 0.99    | 1.0000  | 1.0000  | 1.0000   |

![UNetC3 output](reports/figures/UNet3_validation.png)
![UNetC3 output](reports/figures/UNet5_validation.png)
![UNetC3 output](reports/figures/UNet8_validation.png)
![UNetC3 output](reports/figures/UNet10_validation.png)

### Segformer

#### Train model
```
(seg) user@systemname:project_directory$ python src/models/Segformer/train.py --data_dir /path/to/root/folder --checkpoint_dir /path/to/checkpoint/folder --epochs 50 --batch-size 8 --accelerator gpu --deviceid 1 --patience 10 -mi 10 -vmi 5
```

| Test Metric         | Score                   |
| :---:               | :---:                   | 
| test_loss           | 0.05111832916736603     |
| test_mean_accuracy  | 0.5                     |
| test_mean_iou       | 0.4995069001850329      |

#### Prediction

```
(seg) user@systemname:project_directory$ python src/models/Segformer/predict.py --data_dir /path/to/root/folder --checkpoint_dir /path/to/checkpoints/checkpointfile.ckpt --test_idx 5 --save_dir /path/to/save/folder
```

![Segformer output](reports/figures/segformer_output.png)



## Summary

- State of the art techniques like UNet and Segformer are experimented for Mitosis detection. 
- Based on IOU score, UNet model appears to perform well with a score `0.95`.
- With the IOU score close to 1.0, it is observed that the model fails to predict Mitosis regions instead only background is predicted.
- As per literature, Segformer achieves state-of-the-art performance on multiple common datasets but with this dataset IOU score of only `0.4995069` is achieved.
- To enhance the detection performance of mitosis, efficient preprocessing techniques and a larger dataset of images are essential, considering that mitosis regions are small in size and occur infrequently within each image.   

## Next steps

- To preserve vital information that may have been lost during image resizing, it is worth considering the approach of creating patches using a sliding window technique.
-  