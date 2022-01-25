.. RSE-Distance documentation master file, created by
   sphinx-quickstart on Mon Jan 10 12:40:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RSE-Distance's documentation!
========================================

This is the documentation of the repository RSE-Distance.
This repository contains code for fast numerical computation of the RSE distance.

Contents
---------------------------------
The repository contains three python scripts: **RandomWalkSimulatorCUDA**, **RandomWalkSimulator** and **MeetingTimeEstimator**
Let us briefly describe what each script does:
   * RandomWalkSimulator computes the meeting time of random walks on a graph. It runs on any computer with the appropriate packages installed (see Installation)
   * RandomWalkSimulatorCUDA also computes the meeting time of random walks on a graph. However, it requires GPUs to run (more precisely CudaToolkit 11.3).
   Clearly, RandomWalkSimulatorCUDA is much faster than RandomWalkSimulator, so it is to be preferable if GPUs are available. 
   * MeetingTimeEstimator is a class that makes educated guesses of the meeting times of two walks which have not met, based on the meeting times of walks which have met. 

Each one of these scripts is explained more in detail in the documentation below (see Contents).
A quickstart tutorial is available in the Jupyter Notebook Example.ipynb (see repository)

Installation
---------------------------------
To use the scripts provided here it is necessary to install some packages.
We suggest to install everything using a package installer such as `conda <https://www.anaconda.com/products/individual>`_

Linux and Windows
#########################

After having installed conda, enter the following command in the terminal:

``conda env create -f environment_linux.yml``

This will create a conda environment named **rse-distance** and install all the dependencies to run the version of the scripts which does **not** require GPUs.
If you want to run the version requiring GPUs, you will need to enter the following command:

``conda env create -f environmentCUDA.yml``

This will create a conda environment named **rse-distance-cuda** and install all the dependencies to run the version of the scripts which requires GPUs.
Note that you need to have GPUs on your computer to run this version of the scripts. 

Disclaimer: I have never tried to install the dependencies on Windows. 

MacOS
#########################

After having installed conda, enter the following command in the terminal:

``conda env create -f environment_mac.yml``

This will create a conda environment named **rse-distance** and install all the dependencies to run the version of the scripts which does **not** require GPUs.
MacOS Binaries dont support CUDA, therefore it is not possible to run the GPU version of the script on MacOS.

General remarks on packages 
#########################

The main packages in the environmnet **rse-distance** are `graph_tool <https://graph-tool.skewed.de>`_, `numpy <https://numpy.org>`_, `tqdm <https://github.com/tqdm/tqdm>`_ and `pytorch <https://pytorch.org>`_.. 
The environment **rse-distance-cuda** contains all the packages in the environment **rse-distance** plus `cupy <https://cupy.dev>`_.  
If (for any reason) the installation instructions above do not work for you, you can still try to install the aforementioned packages manually. 

Tutorial
---------------------------------

The Jupiter notebook ``Example.ipynb`` contains a detailed tutorial explaining how to employ the scripts RandomWalkSimulator (or RandomWalkSimulatorCUDA) in order to compute:
   - The RSE distance between two vertices i and j
   - The structural diversity index \Delta(G)


Full code documentation
------------------------------------
Below we provide detailed documentation for the python scripts in the repository RSE-distance



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
