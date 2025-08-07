# Utilities; A set of tools for performing systems analyses of Molecular Dynamics (MD) Simulations

![CI](https://img.shields.io/badge/CI-passing-brightgreen)
[![PyPI version](https://img.shields.io/badge/PyPI--version-inactive.svg)]()
[![Anaconda version](https://img.shields.io/badge/Anaconda--version-inactive.svg)]()
![Downloads](https://img.shields.io/badge/downloads-blank-lightgrey)
[![DOI](https://img.shields.io/badge/DOI--blue)]()

![Alt text](/resources/Pipelineflic.png)

Pictured is a directed graph describing the pipeline for our trajectory analysis. From left to right, we begin with a trajectory file, convert it into networks, which can be represented as either graphs or adjacency matrices. We move forward with the adjacency matrix representations and concatenate each matrix’s rows to create a vector representation of our system at every frame. Next, we vertically concatenate these to create a feature matrix that can be used as input to either K-means or PCA, whose results can then be visualized using the principal components analysis, our MDcircos plots, or replicate maps (which can alternatively represent the raw H-bond counts).

The labs current focus is on computational genetics expirements.
We modify various adjacent codon identites at the A-site and +1 (poised to enter the A site 5'-3')
in order to model the varying behaviors of the CAR interaction surface and how they correllate to varying
translation rate changes.

# The easiest way to get going if your familiar with python developement especially in conda environments is to pip install after forking. I.E.:

```bash

# First fork the repository over 
git clone https://github.com/zeper-eng/workspace.git
cd workspace

#Now from inside of the workspace folder simply pip install!

pip install .

#Regardless of the python environment you should be good to go
pip show workspace
```

# This will eventually be replaced with a proper pip install but we are clearly in development



