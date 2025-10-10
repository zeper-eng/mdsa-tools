.. _subdomain_exploration_from_embeddings:


subdomain_exploration (embedding-space kinetics from embeddings)
=============================================

Use :class:`mdsa_tools.subdomain_explorations` to turn **embedding-space clusters**
Explore (PCA/UMAP + k-means) potential preffered conformational spaces derived from embeddings,and includes cohesion-over-time diagnostics.
lightweight Markov-state analysis: transition matrices, and quick visualizations.

**Note** this module heavily relies on the fact that you have multiple "concatenated" trajectories or replicates but, will then also take that into account and not have cross-replicate boundary counts.


What you get
------------
- A :class:`~mdsa_tools.subdomain_explorations` instance bound to your labels, centers,
  reduced coordinates, and per-replicate frame lengths.
- A **transition probability matrix** ``(n_states+1)×(n_states+1)`` with header row/col.
- **Cohesion over time** (sliding & shrinking windows) as tidy ``pandas.DataFrame``s.


optional
--------

- UMAP/PCA scatter plots colored by clusters or by replicate/frame index.
- **replicate map** images to visualize state visitation by replicate.


Quickstart
----------
Minimal example using two previously saved systems (``.npy`` arrays created earlier
by :class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor` or :mod:`mdsa_tools.Cpptraj_import`).

.. code-block:: python

   from mdsa_tools.Analysis import systems_analysis
   from mdsa_tools.subdomain_exploration_modeler import subdomain_exploration_Modeller
   from mdsa_tools.Viz import visualize_reduction, replicatemap_from_labels
   import numpy as np
   import matplotlib.cm as cm
   import pandas as pd
   import os

   #########################################
   # inputs: two systems of H-bond matrices
   #########################################

   # adjust paths for your machine
   system_gcu_path = "/path/to/redone_unrestrained_CCU_GCU_Trajectory_array.npy"
   system_cgu_path = "/path/to/redone_unrestrained_CCU_CGU_Trajectory_array.npy"

   redone_CCU_GCU_fulltraj = np.load(system_gcu_path, allow_pickle=True)
   redone_CCU_CGU_fulltraj = np.load(system_cgu_path, allow_pickle=True)

   # replicate lengths (frames) used when concatenating replicates per system
   # here: 20 short reps of 80 frames, 10 long reps of 160 frames
   per_rep_lengths = ([80] * 20) + ([160] * 10)

   ########################
   # build embeddings (PCA)
   ########################

   all_systems = [redone_CCU_GCU_fulltraj, redone_CCU_CGU_fulltraj]
   SA = systems_analysis(systems_representations=all_systems,
                         replicate_distribution=per_rep_lengths)
   SA.replicates_to_featurematrix()
   X_pca, _, _ = SA.reduce_systems_representations(method="PCA")

   outdir = "./subdomain_exploration_example_outputs"
   os.makedirs(outdir, exist_ok=True)

   #########################################
   # cluster embedding space → candidate subdomain_exploration
   #########################################

   # pick a K (or call SA.perform_kmeans with max_clusters for an elbow/silhouette search)
   k = 6
   labels, centers = SA.perform_kmeans(data=X_pca, k=k)

   visualize_reduction(
       X_pca,
       color_mappings=labels,
       savepath=os.path.join(outdir, "pca_clusters"),
       cmap=cm.inferno_r,
       title=f"PCA (k={k})"
   )

   #########################################
   # lightweight subdomain_exploration on the clustered embedding
   #########################################

   subdomain_exploration = subdomain_exploration_Modeller(labels, centers, X_pca, frame_scale=per_rep_lengths)

   sliding = subdomain_exploration.evaluate_cohesion_slidingwindow(step_size=20)
   shrinking = subdomain_exploration.evaluate_cohesion_shrinkingwindow(step_size=20)

   sliding.to_csv(os.path.join(outdir, "cohesion_sliding.csv"), index=False)
   shrinking.to_csv(os.path.join(outdir, "cohesion_shrinking.csv"), index=False)

   #########################################
   # optional: replicate/state overview plots
   #########################################

   # replicate map expects labels (optionally with a filler) and per-replicate lengths
   # Here we duplicate per_rep_lengths to match 2 systems concatenated back-to-back.
   two_system_lengths = per_rep_lengths * 2
   replicatemap_from_labels(
       labels,
       two_system_lengths,
       savepath=os.path.join(outdir, "replicate_map_k6"),
       title="Replicate map (k=6)"
   )

Notes
-----
- **Lag is in frames.** If your MD timestep is ``dt`` ps, multiply implied timescales by ``dt`` to convert.
- Labels must be **0-based contiguous** (``0..K-1``) to align with the centers rows and the transition-matrix headers.
- Windowed cohesion never crosses replicate boundaries; replicates shorter than a window step simply don’t contribute to that window. We recommend seperating out different length replicates
- For UMAP, call ``SA.reduce_systems_representations(method="UMAP", n_neighbors=..., min_dist=...)`` and reuse the same workflow.

Where this fits
---------------
- Upstream: :mod:`mdsa_tools.Analysis` produces the feature matrix, embeddings, and clusters.

See also
--------
- :mod:`mdsa_tools.Analysis` — feature matrices, clustering, PCA/UMAP.
- :mod:`mdsa_tools.Viz` — scatter/replicate maps and helpers.
- :mod:`mdsa_tools.Cpptraj_import` — build the same adjacency matrices from cpptraj text tables.
