.. _full_example:

Feature matrix & exploratory clustering
=======================================

Use :class:`mdsa_tools.Analysis.systems_analysis` to assemble feature matrices
from previously saved H-bond adjacency stacks and run **K-means** across a range
of :math:`K`. Then generate low-dimensional embeddings (``PCA`` or ``UMAP``) and,
optionally, perform a second clustering pass directly in embedding space.

What you get
------------
- A :class:`~mdsa_tools.Analysis.systems_analysis` instance bound to your systems.
- Per-K label arrays saved to disk as ``kluster_labels_{K}clust.npy`` plus
  elbow/silhouette score plots and the **optimal K** by each criterion.
- 2D embedding figures (colored by cluster labels) saved to your output folder.
- Optional **embedding-space** cluster labels for finer structure in PCA/UMAP space.

Quickstart
----------
Minimal example using two previously generated systems (``.npy`` arrays created by
:class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor` or :mod:`mdsa_tools.Cpptraj_import`).

If you are unfamiliar with making these, see :ref:`datagen`.

.. code-block:: python

   import os
   import numpy as np
   from mdsa_tools.Analysis import systems_analysis

   #########################################
   # Load systems (each is a list of frames)
   #########################################
   system_a = np.load("/path/to/CCU_GCU_Trajectory_array.npy", allow_pickle=True)
   system_b = np.load("/path/to/CCU_CGU_Trajectory_array.npy", allow_pickle=True)
   all_systems = [system_a, system_b]

   #########################################
   # Build analyzer
   #########################################
   SA = systems_analysis(all_systems)

   #########################################
   # System-level K-means sweep (elbow & silhouette)
   #########################################
   k_labels_sil, k_labels_elbow, centers_sil, centers_elbow = SA.cluster_system_level(
       outfile_path="/path/to/output/syskmeans/",
       max_clusters=25
   )
   print("Clustering successfully completed.")

   #########################################
   # Dimensionality reduction (PCA or UMAP)
   #########################################
   SA.reduce_systems_representations(
       outfile_path="/path/to/output/reduction/test_",
       colormappings=k_labels_sil,         # color points by silhouette-optimal labels
       method="PCA"                         # or "UMAP"
   )
   print("Reduction successful.")

   #########################################
   # Optional: cluster directly in embedding space
   #########################################
   SA.cluster_embeddingspace(
       outfile_path="/path/to/output/cluster_embeddingspace/",
       max_clusters=10,
       elbow_or_sillohuette="sillohuette"   # use "elbow" to switch criterion
   )
   print("Embedding-space clustering successfully completed.")

Notes
-----
- **Inputs:** Each system should be an array-like of shape
  ``(n_frames, n_res+1, n_res+1)``, with the ``0``-th row/col storing **1-based**
  residue IDs. Slice ``[1:, 1:]`` to work with the numeric adjacency submatrix.
- **Saved outputs:** During the K sweep, per-K labels are saved as
  ``kluster_labels_{K}clust.npy`` under ``outfile_path`` along with score plots.
- **Coloring:** Pass any label vector (e.g., silhouette-optimal) via ``colormappings``
  to color points consistently across PCA/UMAP figures.
- **Reproducibility:** Keep inputs/figures in versioned folders so reruns are trivial
  when preprocessing changes.

Where this fits
---------------
- Generate adjacency arrays with :class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor`
  (or build from cpptraj via :mod:`mdsa_tools.Cpptraj_import`), then run clustering here.
- After clustering:
  - Explore embeddings and replicate maps with :mod:`mdsa_tools.Viz`.

See also
--------
- :mod:`mdsa_tools.Data_gen_hbond` — build per-frame H-bond adjacency matrices.
- :mod:`mdsa_tools.Cpptraj_import` — construct the same matrices from cpptraj series tables.
- :meth:`mdsa_tools.Analysis.systems_analysis.cluster_system_level` — K-sweep with elbow/silhouette.
- :meth:`mdsa_tools.Analysis.systems_analysis.reduce_systems_representations` — PCA/UMAP embeddings.
- :meth:`mdsa_tools.Analysis.systems_analysis.cluster_embeddingspace` — clustering in embedding space.
