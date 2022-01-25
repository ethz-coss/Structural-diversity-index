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
The repository contains two python scripts: **RandomWalkSimulatorCUDA** and **RandomWalkSimulator**.
The scripts basically do the same thing, namely, compute the meeting times of random walks on a graph. 
The difference between the scripts is that running RandomWalkSimulatorCUDA requires GPUs, wheras, RandomWalkSimulator runs on any computer with the appropriate packages installed (see requirements.txt). Clearly, RandomWalkSimulatorCUDA is much faster than RandomWalkSimulator, so it is to be preferred if GPUs are available. 

Installation
---------------------------------
To use the scripts provided here it is necessary to install some packages.
We suggest to install everything using a package installer such as `conda <https://www.anaconda.com/products/individual>`_
After having installed conda, enter the following command in the terminal:

```rb
conda env create -f environment.yml
```

This will create a conda environment named **rse-distance** and install all the dependencies to run the version of the scripts which does **not** require GPUs.
If you want to run the version requiring GPUs, you will need to enter the following command:

```rb
conda env create -f environmentCUDA.yml
```

This will create a conda environment named **rse-distance-cuda** and install all the dependencies to run the version of the scripts which requires GPUs.
Note that you need to have GPUs on your computer to run this version of the scripts. 

The main packages in the environmnet **rse-distance** are `graph_tool <https://graph-tool.skewed.de>`_, `numpy <https://numpy.org>`_, `tqdm <https://github.com/tqdm/tqdm>`_ and `pytorch <https://pytorch.org>`_.. 
The environment **rse-distance-cuda** contains all the packages in the environment **rse-distance** plus `cupy <https://cupy.dev>`_. 

Running the code
---------------------------------

The Jupiter notebook ``Example.ipynb`` contains a commented example of how to employ the scripts RandomWalkSimulator (or RandomWalkSimulatorCUDA) in order to compute:
   - The RSE distance between two vertices i and j
   - The structural diversity index \Delta(G)


Documentation
------------------------------------
Below we provide detailed documentation for the python scripts in this repository. 



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
