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
First of all, one must install `graph_tool <https://graph-tool.skewed.de>`_
To install graph_tool with conda it suffices to type into the terminal:

``conda install -c conda-forge graph-tool``

Next one must install `numpy <https://numpy.org>`_, `tqdm <https://github.com/tqdm/tqdm>`_ and `pytorch <https://pytorch.org>`_.
This can be done by typing the following commands into the terminal:

``conda install numpy`` \

``conda install tqdm`` \

``conda install pytorch``

Finally, if you want to use the CUDA version of the python script (i.e. the one using GPUs) it is necessary to install [CuPy](https://cupy.dev). It can be simply done by typing the following command in the terminal:

``conda install -c conda-forge cupy``

Once all packages are installed you can test out the scripts above cloning this Github repository onto your computer and using the Jupiter notebook Example.ipynb  provided below. 

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
