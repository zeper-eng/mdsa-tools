



# __mdsa-tools__ [![Docs Build](https://github.com/zeper-eng/mdsa-tools/actions/workflows/docs.yml/badge.svg?branch=main)](https://mdsa-tools.readthedocs.io/en/latest/)![Last commit](https://img.shields.io/github/last-commit/zeper-eng/mdsa-tools)[![PyPI version](https://img.shields.io/pypi/v/mdsa-tools.svg)](https://pypi.org/project/mdsa-tools/)[![License](https://img.shields.io/pypi/l/mdsa-tools.svg)](https://github.com/zeper-eng/mdsa-tools/blob/main/LICENSE)

Tools for systems-level analysis of Molecular Dynamics (MD) simulations
## Pipeline overview
A set of tools for producing and analyzing systems representations (networks) of Molecular Dynamics (MD) simulations. 

![Pipeline](https://raw.githubusercontent.com/zeper-eng/workspace/main/resources/Pipelineflic.png)

We start from an MD trajectory and generate per-frame interaction networks (graphs/adjacency matrices). Adjacencies are flattened (row-wise) into vectors; stacking these per-frame vectors yields a feature matrix suitable for clustering (e.g., k-means) and dimensionality reduction (PCA/UMAP). Results can be visualized with graphs, scatter plots, MDCcircos plots (residue H-bonding), or replicate maps of frame-level measurements of interest. These clustered states can then serve as candidate substates for constructing and analyzing Markov state models (MSMs), enabling exploration of long-timescale dynamics and transition pathways.

## Install

```bash
pip install mdsa-tools
# Optional:
# pip install "mdsa-tools[docs]"   # if you want to build the docs locally
# pip install "mdsa-tools[examples]"  # if you define this extra for demo deps
```

## Systems Problem Area:

![System panel](https://raw.githubusercontent.com/zeper-eng/workspace/main/resources/PanelA_summerposter.png)

In the Weir Group at Wesleyan University, we perform molecular dynamics (MD) simulations of a ribosomal subsystem to study tuning of protein translation by the CAR interaction surface- a ribosomal interface identified by the lab that interacts with the +1 codon (poised to enter the ribosome A site). Our "computational genetics" research focuses on modifying adjacent codon identities at the A-site and the +1 positions to model how changes at these sites influence the behavior of the CAR surface and corellate with translation rate variations.


## Quickstart example (see examples for more use-cases;contour plots, UMAP, MSM, etc):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/zeper-eng/mdsa-tools/blob/main/notebooks/Quick_Start.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](
https://mybinder.org/v2/gh/zeper-eng/mdsa-tools/HEAD?labpath=notebooks/Quick_Start.ipynb)
[![nbviewer](https://img.shields.io/badge/View%20Notebook-nbviewer-blue)](
https://nbviewer.org/github/zeper-eng/mdsa-tools/blob/main/notebooks/Quick_Start.ipynb)

```python
from mdsa_tools.Data_gen_hbond import trajectory
from mdsa_tools.Analysis import systems_analysis
import numpy as np

# In house test with our own trajectories

system_one_topology = '/Users/luis/Desktop/workspace/PDBs/5JUP_N2_CGU_nowat.prmtop'
system_one_trajectory = '/Users/luis/Desktop/workspace/PDBs/CCU_CGU_10frames.mdcrd'

system_two_topology = '/Users/luis/Desktop/workspace/PDBs/5JUP_N2_GCU_nowat.prmtop'
system_two_trajectory = '/Users/luis/Desktop/workspace/PDBs/CCU_GCU_10frames.mdcrd'

test_trajectory_one = trajectory(
    trajectory_path=system_one_trajectory,
    topology_path=system_one_topology
)
test_trajectory_two = trajectory(
    trajectory_path=system_two_trajectory,
    topology_path=system_two_topology
)

# now that they're loaded, make system representations
test_system_one_ = test_trajectory_one.create_system_representations()
test_system_two_ = test_trajectory_two.create_system_representations()

# save for reuse
np.save('/Users/luis/Desktop/workspacetwo/example_systems/test_system_one', test_system_one_)
np.save('/Users/luis/Desktop/workspacetwo/example_systems/test_system_two', test_system_two_)

###
### Analysis
###
analyzer = systems_analysis([sys1, sys2])

# Clustering
sil_labels, elbow_labels, sil_centers, elbow_centers = analyzer.cluster_system_level(
    outfile_path="out/syskmeans/", max_clusters=25
)
print("Clustering successfully completed.")

# Dimensional reduction (PCA or UMAP); color by cluster labels
analyzer.reduce_systems_representations(
    outfile_path="out/PCA/test_", 
    method="PCA",
    colormappings=sil_labels
)
print("PCA reduction successful.")
```




