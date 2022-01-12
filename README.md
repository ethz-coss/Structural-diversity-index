# RSE-Distance
This repository contains code for fast numerical computation of the RSE distance.

## Contents
The repository contains two python scripts: **RandomWalkSimulatorCUDA** and **RandomWalkSimulator**.
The scripts basically do the same thing, namely, compute the meeting times of random walks on a graph. 
The difference between the scripts is that running RandomWalkSimulatorCUDA requires GPUs, wheras, RandomWalkSimulator runs on any computer with the appropriate packages installed (see requirements.txt). Clearly, RandomWalkSimulatorCUDA is much faster than RandomWalkSimulator, so it is to be preferred if GPUs are available. 

## Installation

To use the scripts provided here it is necessary to install some packages.
We suggest to install everything using a package installer such as [conda](https://www.anaconda.com/products/individual). 
First of all, one must install [graph_tool](https://graph-tool.skewed.de)
To install graph_tool with conda it suffices to type into the terminal:

```rb
conda install -c conda-forge graph-tool
```

Next one must install [numpy](https://numpy.org), [tqdm](https://github.com/tqdm/tqdm) and [pytorch](https://pytorch.org). This can be done by typing the following commands into the terminal:

```rb
conda install numpy
conda install tqdm
conda install pytorch
```

Finally, if you want to use the CUDA version of the python script (i.e. the one using GPUs) it is necessary to install [CuPy](https://cupy.dev). It can be simply done by typing the following command in the terminal:

```rb
conda install -c conda-forge cupy
```

Once all packages are installed you can test out the scripts above cloning this Github repository onto your computer and using the Jupiter notebook Example.ipynb  provided below. 

## Running the code

The Jupiter notebook **Example.ipynb** contains a commented example of how to employ the scripts RandomWalkSimulator (or RandomWalkSimulatorCUDA) in order to compute:
* The RSE distance between two vertices i and j
* The structural diversity index \Delta(G)



