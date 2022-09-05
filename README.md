# RSE-Distance
This repository contains code for fast numerical computation of the structural diversity index

## Contents
The repository contains four python scripts:**MeetingTimesUI**, **RandomWalkSimulatorCUDA**, **RandomWalkSimulator** and **MeetingTimeEstimator**
Let us briefly describe what each script does:
   * MeetingTimeUI provides a user interface for the scripts
   * RandomWalkSimulator computes the meeting time of a random walk on a graph. 
   * RandomWalkSimulatorCUDA computes the meeting time of random walks on a graph using CUDA and GPUs (much faster for large graphs). It requires CudaToolkit 11.3 to run.
   * MeetingTimeEstimator is a class that makes educated guesses of the meeting times of two walks which have not met, based on the meeting times of walks which have met. 

Each one of these scripts is explained more in detail in the documentation provided [here](https://rse-distance.readthedocs.io).
If you are interested in a **quick start tutorial** see the section **Tutorial** below.

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

## Tutorial

The Jupiter notebook **Example.ipynb** contains a detailed tutorial explaining how to employ the scripts RandomWalkSimulator (or RandomWalkSimulatorCUDA) in order to compute:
* The RSE distance between two vertices i and j
* The structural diversity index \Delta(G)

## Extending the code

If you are interested in extending or simply playing around with the code, I have created a detailed documentation with ReadTheDocs which is available [here](https://rse-distance.readthedocs.io). Have fun!



