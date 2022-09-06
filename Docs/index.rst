.. Structural diversity index documentation master file, created by
   sphinx-quickstart on Mon Sep  5 18:42:11 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Structural diversity index
=====================================
This is the documentation for the repository `Structural-diversity-index <https://pypi.org/project/structural-diversity-index/>`_.
This repository contains code for fast numerical computation of the structural diversity index.

Contents
----------------------
The repository contains four python scripts: **MeetingTimesUI**, **RandomWalkSimulatorCUDA**, **RandomWalkSimulator** and **MeetingTimeEstimator**.
Here is a brief description:

   * MeetingTimeUI provides a user interface for the scripts
   * RandomWalkSimulator computes the meeting time of a random walk on a graph.
   * RandomWalkSimulatorCUDA computes the meeting time of random walks on a graph using CUDA and GPUs (much faster for large graphs). It requires Cudatoolkit to run.
   * MeetingTimeEstimator is a class that makes educated guesses of the meeting times of two walks which have not met, based on the meeting times of walks which have met.

If you are interested in a **quick start tutorial** see the section **Tutorial** below.

Installation
-----------------------

The scripts are provided in the form of a python package called `structural_diversity_index <https://pypi.org/project/structural-diversity-index/>`_.
To install the package and its dependencies type into the terminal

``pip install structural_diversity_index==0.0.3``

This will install the 0.0.3 version (latest) of the package in your python packages directory.

**WARNING**: Installing the package via pip will allow **NOT** you to use the scripts that run computations on GPUs.
See below for details of how to run the scripts computing on GPUs.

Installation for GPUs
___________________________________________
If you are not interested in running computations on GPUs you can ignore this section.

Installing the structural_diversity_index package via pip does not enable you to run computations on GPUs.
The reason is that the Cudatoolkit cannot be installed by pip (because it is not a python package).

To circumvent this issue one can use a package installer such as `conda <https://www.anaconda.com/products/individual>`_.
Once you have installed conda on your computer, download the file **environment.yml** from the `GitHub <https://github.com/ethz-coss/Structural-diversity-index>`_.
In the terminal, go to the directory containing the environment.yml file you downloaded and type:

``conda env create -f environment.yml``

This will create a conda environment called **sd_index** and install all the dependencies necessary to computations on GPUs.
Now you can set on_cuda=True (see Examples.ipynb in `GitHub <https://github.com/ethz-coss/Structural-diversity-index>`_) and computations will run on GPUs.

Tutorial
------------------------
The Jupyter notebook **Example.ipynb** contains a detailed tutorial explaining how to use the package structural_diversity_index.
You can find it `here <https://github.com/ethz-coss/Structural-diversity-index>`_.


Welcome to Structural diversity index's documentation!
======================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
