# Structural diveristy index
This repository contains code for fast numerical computation of the structural diversity index.

## Contents
The repository contains four python scripts: **MeetingTimesUI**, **RandomWalkSimulatorCUDA**, **RandomWalkSimulator** and **MeetingTimeEstimator**
Here is a brief description:
   * MeetingTimeUI provides a user interface for the scripts.
   * RandomWalkSimulator computes the meeting time of a random walk on a graph. 
   * RandomWalkSimulatorCUDA computes the meeting time of random walks on a graph using CUDA and GPUs (much faster for large graphs). It requires Cudatoolkit to run.
   * MeetingTimeEstimator is a class that makes educated guesses of the meeting times of two walks which have not met, based on the meeting times of walks which have met. 

Each script is described in detail in the documentation provided [here](https://rse-distance.readthedocs.io).
If you are interested in a **quick start tutorial** see the section **Tutorial** below.

## Installation

The scripts are provided in the form of a python package called [structural_diversity_index](https://pypi.org/project/structural-diversity-index/).
To install the package and its dependencies type into the terminal
```rb
pip install structural_diversity_index==0.0.5
```
This will install the 0.0.5 version (latest) of the package in your python packages directory.

**WARNING**: Installing the package via pip will allow **NOT** you to use the scripts that run computations on GPUs.
See below for details of how to run the scripts computing on GPUs.
### Installation for GPUs
If you are not interested in running computations on GPUs you can ignore this section.

Installing the structural_diversity_index package via pip does not enable you to run computations on GPUs.
The reason is that the Cudatoolkit cannot be installed by pip (because it is not a python package).

To circumvent this issue one can use a package installer such as [conda](https://www.anaconda.com/products/individual).
Once you have installed conda on your computer, download the file **environment.yml** from the [GitHub](https://github.com/ethz-coss/Structural-diversity-index).
In the terminal, go to the directory containing the environment.yml file you downloaded and type:

```rb
conda env create -f environment.yml
```

This will create a conda environment called **sd_index** and install all the dependencies necessary to computations on GPUs.
Now you can set on_cuda=True (see Examples.ipynb in [GitHub](https://github.com/ethz-coss/Structural-diversity-index)) and computations will run on GPUs. 

## Tutorial

The Jupyter notebook **Example.ipynb** (available [here](https://github.com/ethz-coss/Structural-diversity-index)) contains a detailed tutorial explaining how to use the package structural_diversity_index.

## Extending the code

If you are interested in extending, modifying or simply playing around with the code, I have created a detailed documentation with ReadTheDocs which is available [here](https://rse-distance.readthedocs.io). 
Have fun!




