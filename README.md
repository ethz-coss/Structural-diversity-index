# Structural diversity index

Welcome to the repository structural-diversity-index.
This repository contains code for fast numerical computation of the structural diversity index.
This code is available in a python package that you can find [here](https://pypi.org/project/structural-diversity-index/).

## Contents
The repository contains four python scripts: **MeetingTimesUI**, **RandomWalkSimulatorCUDA**, **RandomWalkSimulator** and **MeetingTimeEstimator**.
Here is a brief description:

   * MeetingTimeUI provides a user interface for the scripts
   * RandomWalkSimulator computes the meeting time of a random walk on a graph.
   * RandomWalkSimulatorCUDA computes the meeting time of random walks on a graph using CUDA and GPUs (much faster for large graphs). It requires Cudatoolkit to run.
   * MeetingTimeEstimator is a class that makes educated guesses of the meeting times of two walks which have not met, based on the meeting times of walks which have met.

If you are interested in a **quick start tutorial** see the section **Tutorial** below.

## Installation

The scripts are provided in the form of a python package called [structural_diversity_index](https://pypi.org/project/structural-diversity-index/).
To install the package and its dependencies you should create a [python virtual environment](https://docs.python.org/3/library/venv.html).
A detailed tutorial about virtual environments is available [here](https://docs.python.org/3/tutorial/venv.html). 
However, if you are in a hurry you can just open a terminal window and type

```
python3 -m venv sdi_venv
```
This creates a virtual environment called **sdi_venv** in your current directory.
Next, activate the virtual environment. This is done on Unix or macOS by typing into the terminal
```
source sdi_venv/bin/activate
```
On Windows you should type
```
sdi_venv\Scripts\activate.bat
```
Once you created and activated the virtual environment you can install the structural_diversity_index package by typing into the terminal
```
pip install structural_diversity_index
```

This will install the latest version  of the package in the virtual environment.

**WARNING 1**: Installing the package via pip will allow **NOT** you to use the scripts that run computations on GPUs.
See below for details of how to run the scripts computing on GPUs.

**WARNING 2**: Installing the package into an existing virtual environment can cause the code to break due to conflicts with already
existing dependencies. For this reason we advise to create a new virtual environment. 

### Installation for GPUs

If you are not interested in running computations on GPUs you can ignore this section.

Installing the structural_diversity_index package via pip does not enable you to run computations on GPUs.
The reason is that pip cannot install Cudatoolkit (because it is not a python package).

To circumvent this issue one can use a package installer such as [conda](https://www.anaconda.com/products/individual).
Once you have installed conda on your computer, download the file **environment.yml** from the [GitHub](https://github.com/ethz-coss/Structural-diversity-index).
In the terminal, go to the directory containing the environment.yml file you downloaded and type:

```
conda env create -f environment.yml
```

This will create a conda environment called **sd_index** and install all the dependencies necessary to computations on GPUs.
Now you can set the flag on_cuda to True (see Examples.ipynb on [GitHub](https://github.com/ethz-coss/Structural-diversity-index>)) and computations will run on GPUs.

## Tutorial
The Jupyter notebook **Example.ipynb** contains a detailed tutorial explaining how to use the package structural_diversity_index.
You can find it [here](https://github.com/ethz-coss/Structural-diversity-index).


## Extending the code
If you are interested in extending or simply playing around with the code, I have created a detailed documentation with ReadTheDocs which is available [here](https://rse-distance.readthedocs.io). 
Have fun!



