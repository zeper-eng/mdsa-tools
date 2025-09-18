.. _datagen:

Data generation (H-bond matrices)
=================================

Use :class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor` to turn trajectories
(+ a topology) into per-frame residue×residue hydrogen-bond adjacency matrices.
These arrays feed directly into the rest of the pipeline (clustering, PCA/UMAP,
MSM bits).

What you get
------------
- For each trajectory: an array of shape ``(n_frames, n_res+1, n_res+1)``.
- Row/col ``0`` store **1-based** residue indices for convenience.
- The submatrix ``[1:, 1:]`` is the numeric adjacency (typically 0/1 or counts).

Quickstart
----------
Minimal example with two systems (adjust paths for your machine).

.. code-block:: python

   from mdsa_tools.Data_gen_hbond import TrajectoryProcessor as tp
   import numpy as np
   import os

   #########################################
   # in-house test with our own trajectories
   #########################################

   # paths (AMBER prmtop + mdcrd in this example)
   system_one_topology = "/path/to/5JUP_N2_CGU_nowat.prmtop"
   system_one_trajectory = "/path/to/CCU_CGU_10frames.mdcrd"

   system_two_topology = "/path/to/5JUP_N2_GCU_nowat.prmtop"
   system_two_trajectory = "/path/to/CCU_GCU_10frames.mdcrd"

   # construct processors
   traj_one = tp(trajectory_path=system_one_trajectory, topology_path=system_one_topology)
   traj_two = tp(trajectory_path=system_two_trajectory, topology_path=system_two_topology)

   # build per-frame adjacency matrices
   system_one = traj_one.create_system_representations()
   system_two = traj_two.create_system_representations()

   print(system_one[0].shape)  # (n_res+1, n_res+1)
   print(system_two[0].shape)

   # save for later steps in the pipeline
   outdir = "/path/to/example_systems"
   os.makedirs(outdir, exist_ok=True)
   np.save(os.path.join(outdir, "test_system_one.npy"), system_one)
   np.save(os.path.join(outdir, "test_system_two.npy"), system_two)

Notes
-----
- Inputs assume an **MDTraj-readable topology** (e.g., AMBER ``.prmtop``) and
  a compatible trajectory (e.g., ``.mdcrd``, ``.xtc``, ``.dcd``).
- The adjacency matrices are **not symmetrized** here unless your definition implies it.
  Post-process as needed (e.g., take the upper triangle, average, etc.).
- The leading index row/col keeps downstream tooling simple; slice ``[1:, 1:]``
  whenever you just want the numeric part.

Where this fits
---------------
- Feed the saved arrays into :mod:`mdsa_tools.Analysis` for feature-matrix
  construction and clustering.
- Reduce and plot with :mod:`mdsa_tools.Viz`.
- If your H-bond data comes from cpptraj text outputs instead of trajectories,
  see :mod:`mdsa_tools.Cpptraj_import`.



See also
--------
- :mod:`mdsa_tools.Analysis` — clustering, PCA/UMAP, MSM helpers.
- :mod:`mdsa_tools.Viz` — quick plots for embeddings and replicate maps.
- :mod:`mdsa_tools.Cpptraj_import` — build the same matrices from cpptraj series tables.
