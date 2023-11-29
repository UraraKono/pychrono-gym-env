# pychrono-gym-env
Open AI Gym-like environment for Pychrono.

## Installation of chrono

### Recommended way
Suppose you have CUDA version 12.1 or newer.
1. Clone to the github repo chrono-tunerINN

2. Try creating an environment from this yaml file in this repo using:
```bash
conda env create -f environment.yml -n chrono
```

3. Download this version of pychrono: https://anaconda.org/projectchrono/pychrono/8.0.0/download/linux-64/pychrono-8.0.0-py39_1.tar.bz2

4. Once it successfully creates the environment, activate it and do:
```bash
conda install pychrono-8.0.0-py39_1.tar.bz2
```

### Create empty conda environment first
You can manually install the packages too.
Please install Pychrono at the end as mentioned in the step 6.

1. Create conda environment with python 3.9 (As of November 2023, Numba doesn't support newer python version.)
```bash
conda create --name chrono_env python=3.9
```

2. Install pytorch and pytorch-cuda that corresponds to your CUDA version. (Installing torchvision caused dependencies issues as of Nov 2023 with) following [Pytorch website](https://pytorch.org/get-started/locally/).

3. Install the following packages in this order.
```bash
conda install pyyaml matplotlib gpytorch pyglet
conda install scipy
conda install -c conda-forge mkl=2020
conda install -c conda-forge irrlicht=1.8.5
conda install -c conda-forge pythonocc-core=7.4.1
conda install cuda-toolkit
conda install -c conda-forge glfw
pip install numba
pip install cvxpy
```
4. Install the packages that you want additionally.

5. Download this version of pychrono: https://anaconda.org/projectchrono/pychrono/8.0.0/download/linux-64/pychrono-8.0.0-py39_1.tar.bz2

6. *After you installed all the necessary packages, finally* install Pychrono.
'''bash
conda install pychrono-8.0.0-py39_1.tar.bz2
'''

## Code description
* main.py: Run the simulation on the maps from maps directory

* chrono_env/environment.py: This file defines env.make() and env.step()

* EGP: The files imported from [this repo](https://github.com/mlab-upenn/multisurface-racing). It has a pure pursuit planner and SaoPaulo map.

* demo_VEH_HMMWV_circle_comment.py: Adding a bit of comments to one of Pychrono's demo file demo_VEH_HMMWV_circle.py. This file doesn't use chrono_env but might be helpful to get the overview of Pychrono's usage.

For a bit more explanation of Pychrono, [this](https://docs.google.com/document/d/1A_kbgo-aT6AN3jz1o6GlgURnv0Lm8bGczRchwqg7-yI/edit?usp=sharing) might be useful.





