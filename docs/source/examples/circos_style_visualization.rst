.. _analysis:

Analysis & MDcircos (PCA-weighted contacts)
===========================================

Use :class:`mdsa_tools.Analysis.systems_analysis` to build feature matrices from
saved H-bond adjacency arrays and compute PCA weightings. Then visualize the
most salient residue–residue contacts with :mod:`mdsa_tools.Viz`’s MDcircos helpers.

What you get
------------
- A :class:`~mdsa_tools.Analysis.systems_analysis` instance bound to your list of systems.
- A PCA-ranked weights table (``pandas.DataFrame``) with columns like
  ``Comparisons``, ``PC1_Weights``, and ``PC1_magnitude``, sorted by importance.
- Circos-style images that highlight contact weights overall or split by
  **positive vs. negative** loadings on the first principal component.

Quickstart
----------
Minimal example using two previously generated systems (``.npy`` arrays created by
:class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor` or :mod:`mdsa_tools.Cpptraj_import`).

.. code-block:: python

   import os
   import numpy as np
   from mdsa_tools.Analysis import systems_analysis
   from mdsa_tools.Viz import create_MDcircos_from_weightsdf

   #########################################
   # Load systems (each is a list of frames
   #########################################

   system_a = np.load("/path/to/CCU_GCU_Trajectory_array.npy", allow_pickle=True)
   system_b = np.load("/path/to/CCU_CGU_Trajectory_array.npy", allow_pickle=True)

   # Each system should be an array-like of shape:
   # (n_frames, n_res+1, n_res+1), where the leading row/col store 1-based residue IDs.
   all_systems = [system_a, system_b]

   #########################################
   # Build analyzer and get weights
   #########################################
   SA = systems_analysis(all_systems)
   PCA_ranked_weights = SA.create_PCA_ranked_weights()  # DataFrame with PC loadings

   #########################################
   # One-shot MDcircos of top-weighted pairs
   #########################################
   outdir = "/path/to/output/circos"
   os.makedirs(outdir, exist_ok=True)
   create_MDcircos_from_weightsdf(PCA_ranked_weights, os.path.join(outdir, "PC1_overall"))

In our case, PC1 clearly separates systems (e.g., one system loads positively and the
other negatively). In that case, we wanted to plot **only** positive or **only**
negative PC1 weights:

.. code-block:: python

   from mdsa_tools.Viz import create_MDcircos_from_weightsdf

   # Keep only the columns we need for filtering/plotting
   PC1_df = PCA_ranked_weights[["Comparisons", "PC1_magnitude", "PC1_Weights"]]

   pos = PC1_df[PC1_df["PC1_Weights"] > 0][["Comparisons", "PC1_magnitude"]]
   neg = PC1_df[PC1_df["PC1_Weights"] < 0][["Comparisons", "PC1_magnitude"]]

   create_MDcircos_from_weightsdf(pos, os.path.join(outdir, "PC1_positive"))
   create_MDcircos_from_weightsdf(neg, os.path.join(outdir, "PC1_negative"))

Notes
-----
- **Inputs:** Each system is a per-frame H-bond adjacency stack of shape
  ``(n_frames, n_res+1, n_res+1)``, with the ``0``-th row/col storing **1-based**
  residue indices. Slice ``[1:, 1:]`` if you only need the numeric adjacency.
- **Weight scaling:** MDcircos helpers may apply **min–max scaling** to magnitudes
  for legibility, which is helpful even if PCA weights are already normalized.
- **Filtering:** The ``PCA_ranked_weights`` table is typically sorted by absolute
  contribution. You can subset by sign (positive/negative) or threshold by
  ``PC1_magnitude`` before plotting.
- **Reproducibility:** Keep your saved arrays and output figures in versioned folders
  so you can re-run the analysis when trajectory preprocessing changes.

Where this fits
---------------
- Generate the adjacency arrays with :class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor`
  (or build from cpptraj tables via :mod:`mdsa_tools.Cpptraj_import`), then pass them here.
- After identifying key contacts with PCA/MDcircos, you can:
  - Cluster or label states with :mod:`mdsa_tools.Analysis`.
  - Reduce and visualize embeddings with :mod:`mdsa_tools.Viz`.
  - Feed features into MSM workflows (see :mod:`mdsa_tools.msm_modeler`).

See also
--------
- :mod:`mdsa_tools.Data_gen_hbond` — build per-frame H-bond adjacency matrices.
- :mod:`mdsa_tools.Cpptraj_import` — construct the same matrices from cpptraj series tables.
- :meth:`mdsa_tools.Analysis.systems_analysis.create_PCA_ranked_weights` — ranked PC loadings.
- :func:`mdsa_tools.Viz.create_MDcircos_from_weightsdf` — quick MDcircos from a weights DataFrame.
