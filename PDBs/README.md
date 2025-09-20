# PDBs

This folder contains **small test PDB-derived files** (prmtop/mdcrd) used throughout the documentation and quickstart examples.  
They serve as lightweight input systems so users can try out the pipeline without needing to generate full molecular dynamics trajectories.

- `5JUP_N2_CGU_nowat.prmtop` – example AMBER topology file.  
- `CCU_CGU_10frames.mdcrd` – short trajectory (10 frames).  
- `5JUP_N2_GCU_nowat.prmtop` – second topology variant.  
- `CCU_GCU_10frames.mdcrd` – matching short trajectory.  

---

## Earlier Quickstart with PDBs


```python
from mdsa_tools.Data_gen_hbond import TrajectoryProcessor as tp
from mdsa_tools.Analysis import systems_analysis
from mdsa_tools.Viz import visualize_reduction
import numpy as np, os, matplotlib.cm as cm

###
### Example datagen using PDBs
###
system_one_top = "./PDBs/5JUP_N2_CGU_nowat.prmtop"
system_one_traj = "./PDBs/CCU_CGU_10frames.mdcrd"

system_two_top = "./PDBs/5JUP_N2_GCU_nowat.prmtop"
system_two_traj = "./PDBs/CCU_GCU_10frames.mdcrd"

traj_one = tp(trajectory_path=system_one_traj, topology_path=system_one_top)
traj_two = tp(trajectory_path=system_two_traj, topology_path=system_two_top)

system_one = traj_one.create_system_representations()
system_two = traj_two.create_system_representations()

np.save("system_one.npy", system_one)
np.save("system_two.npy", system_two)

###
### Analysis
###
all_systems = [system_one, system_two]
analyzer = systems_analysis(all_systems)

analyzer.replicates_to_featurematrix()
labels, _, _, _ = analyzer.cluster_system_level(outfile_path="./test_", max_clusters=5)
X_pca, weights, var = analyzer.reduce_systems_representations(method="PCA")

###
### Visualization
###
visualize_reduction(
    X_pca,
    color_mappings=labels,
    savepath="./PCA_",
    cmap=cm.plasma_r
)
