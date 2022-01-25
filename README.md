# RSE-Distance
This repository contains code for fast numerical computation of the RSE distance.

## Contents
The repository contains two python scripts: **RandomWalkSimulatorCUDA** and **RandomWalkSimulator**.
The scripts basically do the same thing, namely, compute the meeting times of random walks on a graph. 
The difference between the scripts is that running RandomWalkSimulatorCUDA requires GPUs, wheras, RandomWalkSimulator runs on any computer with the appropriate packages installed. Clearly, RandomWalkSimulatorCUDA is much faster than RandomWalkSimulator, so it is to be preferred if GPUs are available. 

## Installation

To use the scripts provided here it is necessary to install some packages.
We suggest to install everything using a package installer such as [conda](https://www.anaconda.com/products/individual).

### Linux and Windows
After having installed conda, enter the following command in the terminal:

```rb
conda env create -f environment_linux.yml
```

This will create a conda environment named **rse-distance** and install all the dependencies to run the version of the scripts which does **not** require GPUs.
If you want to run the version requiring GPUs, you will need to enter the following command:

```rb
conda env create -f environmentCUDA.yml
```

This will create a conda environment named **rse-distance-cuda** and install all the dependencies to run the version of the scripts which requires GPUs.
Note that you need to have GPUs on your computer to run this version of the scripts. 

Disclaimer: I have never tried to install the dependencies on Windows. 

### MacOS
After having installed conda, enter the following command in the terminal:

```rb
conda env create -f environment_mac.yml
```

This will create a conda environment named **rse-distance** and install all the dependencies to run the version of the scripts which does **not** require GPUs.
MacOS Binaries dont support CUDA, therefore it is not possible to run the GPU version of the script on MacOS.

### General remarks on packages 

The main packages in the environmnet **rse-distance** are [graph_tool](https://graph-tool.skewed.de), [numpy](https://numpy.org), [tqdm](https://github.com/tqdm/tqdm) and [pytorch](https://pytorch.org). The environment **rse-distance-cuda** contains all the packages in the environment **rse-distance** plus [cupy](https://cupy.dev). If (for any reason) the installation instructions above do not work for you, you can still try to install the aforementioned packages manually. 

## Running the code

The Jupiter notebook **Example.ipynb** contains a commented example of how to employ the scripts RandomWalkSimulator (or RandomWalkSimulatorCUDA) in order to compute:
* The RSE distance between two vertices i and j
* The structural diversity index \Delta(G)

## Extending the code

If you are interested in extending or simply playing around with the code, I have created a detailed documentation with ReadTheDocs which is available [here](rse-distance.readthedocs.io). Have fun!



