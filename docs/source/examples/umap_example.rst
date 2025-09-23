.. _examples_index:

UMAP embeddings (per-frame labelling)
=====================================
Use :class:`mdsa_tools.Analysis.systems_analysis` to flatten per-frame H-bond
adjacency matrices into a feature matrix, then run a non-linear reduction with
**UMAP**. In this example, we color the embedding by a **per-replicate frame
index** so you can see temporal progression (“crawl”) within each replicate.

What you get
~~~~~~~~~~~~

- A :class:`~mdsa_tools.Analysis.systems_analysis` instance bound to your systems.
- A 2-D UMAP embedding (``(n_frames_total, 2)``).
- A saved figure colored by the per-replicate frame index.

Quickstart
~~~~~~~~~~

Minimal example using two pre-generated systems (``.npy`` arrays made by
:class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor` or
:mod:`mdsa_tools.Cpptraj_import`). If you’re unfamiliar with producing these arrays,
see :ref:`datagen`.

.. code-block:: python

   import os
   import numpy as np
   from mdsa_tools.Analysis import systems_analysis
   from mdsa_tools.Viz import visualize_reduction 

   #####################################
   # Load systems (stacks of frames)
   #####################################

   # Each system: (n_frames, n_res+1, n_res+1). Row/col 0 stores 1-based residue IDs.
   sys_A = np.load(
       "/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy",
       allow_pickle=True,
   )
   sys_B = np.load(
       "/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy",
       allow_pickle=True,
   )
   all_systems = [sys_A, sys_B]

   #####################################
   # Build analyzer & feature matrix
   #####################################

   SA = systems_analysis(all_systems)
   X = SA.replicates_to_featurematrix()  # (sum(n_frames), n_features_uppertri)

   #####################################
   # Per-replicate frame-index labels
   #####################################

   # Example: for each replicate, 20×80 frames then 10×160 frames; duplicated for two systems.
   frame_blocks = (([80] * 20) + ([160] * 10)) * 2
   per_rep_timeframe = np.concatenate([np.arange(0, n, 1) for n in frame_blocks])

   assert per_rep_timeframe.shape[0] == X.shape[0], \
       "Frame-label vector must match total frames across systems."

   #####################################
   # Run UMAP (non-linear reduction)
   #####################################

   umap_xy = SA.reduce_systems_representations(
       feature_matrix=X,
       method="UMAP",
       n_components=2,
       n_neighbors=50,  # try 15–200
       min_dist=0.1,    # try 0.0–0.8
   )

   #####################################
   # Plot & save (using project helper)
   #####################################

   outdir = "/Users/luis/Downloads/embeddings"
   os.makedirs(outdir, exist_ok=True)
   outfile = os.path.join(outdir, "umap_per_replicate_frame.png")

   visualize_reduction(
       umap_xy,
       labels=per_rep_timeframe,
       title="UMAP of per-frame features (colored by per-replicate frame index)",
       savepath=outfile,
   )
Notes
~~~~~

- **Input format:** Each system is a per-frame adjacency stack shaped
  ``(n_frames, n_res+1, n_res+1)`` with **1-based** residue indices in the leading row/col.
  Slice ``[:, 1:, 1:]`` if you only need the numeric adjacency.
- **Feature matrix:** :meth:`~mdsa_tools.Analysis.systems_analysis.replicates_to_featurematrix`
  flattens each frame’s **upper triangle** so rows correspond to frames and columns to
  unique residue–residue pairs.
- **Labels:** ``per_rep_timeframe`` must be the same length as the total number of frames
  across all systems (concatenation order should match your feature stacking).
- **Tuning UMAP:**
  
  * ``n_neighbors`` ↑ → more global structure; ↓ → more local detail.
  * ``min_dist`` ↓ → tighter clusters; ↑ → more spread.
  
  Consider standardizing features first if magnitudes vary widely.
- **Reproducibility:** Set ``random_state`` on UMAP if you need deterministic layouts
  (e.g., ``umap.UMAP(random_state=0, ...)`` inside
  :meth:`~mdsa_tools.Analysis.systems_analysis.reduce_systems_representations`).

Where this fits
~~~~~~~~~~~~~~~

After :ref:`datagen`, use UMAP/PCA to visualize conformational heterogeneity. Reuse
the same feature matrix for clustering
(:meth:`~mdsa_tools.Analysis.systems_analysis.perform_kmeans`) or MDcircos
weighting downstream. Pair with replicate maps to connect embedding regions to
contact-pattern dynamics.

See also
~~~~~~~~

- :mod:`mdsa_tools.Data_gen_hbond` — build per-frame H-bond adjacency matrices.
- :mod:`mdsa_tools.Cpptraj_import` — construct matrices from cpptraj series tables.
- :meth:`mdsa_tools.Analysis.systems_analysis.replicates_to_featurematrix` — feature preparation.
- :mod:`mdsa_tools.Viz` *(optional)*: ``visualize_reduction`` scatter helper.