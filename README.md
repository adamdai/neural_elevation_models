# Neural Elevation Models (NEMo) for Terrain Mapping and Planning

Code for Neural Elevation Models (NEMo), and framework for terrain mapping and path planning. 
This repo contains code for loading trained NEMos and performing path planning on them.
The code for NEMo training can be found at: https://github.com/Stanford-NavLab/nerfstudio/tree/adam/terrain

<br>
<p align="center">
<img src='results/overview_horizontal.png' width="900">
</p>

## Setup

Clone the GitHub repository:

    git clone https://github.com/adamdai/neural_elevation_models.git

Create and activate conda environment:

    conda create -n nemo python=3.8   
    conda activate nemo
    
Install dependencies:

    pip install -r requirements.txt
    pip install -e .

Install pytorch (https://pytorch.org/get-started/locally/), may need to reboot after first line

    sudo apt install -y nvidia-cuda-toolkit
    pip3 install torch torchvision torchaudio

Install tiny-cuda-nn:

    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

Install GDAL (for working with `.tif` DEMs):

    conda install -c conda-forge gdal

(Developed and tested with Ubuntu 20.04 and Windows 10)

## Data

Download the data folder from [this link](https://drive.google.com/drive/folders/1Raa2Z5ZtL9PDKfbaokhcdgk0SGlL-kAl?usp=sharing) and place it in the repo.
```
data/
|-- lunar/
|-- kt22/
|   |-- colmap_points3D.txt
|-- redrocks/
|   |-- DEM32-DroneMapper.tif
|   |-- colmap_points3D.txt
```
These files are used for DEM comparison to COLMAP and ground truth.

## Models

Weights of trained height networks for the KT-22 and Red Rocks scenes can be found under the `models` folder. 


## Path Planning

The notebook `height_net.ipynb` loads a trained NEMo (KT-22 or Red Rocks), and performs path planning via A* initialization then continuous path optimization.
