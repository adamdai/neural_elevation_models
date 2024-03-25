# Neural Elevation Models (NEMo) for Terrain Mapping and Planning

Code for Neural Elevation Models (NEMo), and framework for terrain mapping and path planning. 
This repo contains code for loading trained NEMos and performing path planning on them.
The code for NEMo training can be found at: https://github.com/Stanford-NavLab/nerfstudio/tree/adam/terrain

{% include_relative results/path.html}

Developed and tested on Ubuntu 20.04 and Windows 10.

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


## Models

Weights from trained models can be found under the models folder. Currently for the KT-22 and Red Rocks scenes.


## Path Planning

The notebook `height_net.ipynb` loads a trained NEMo (KT-22 or Red Rocks), and performs path planning via A* initialization then continuous path optimization.
