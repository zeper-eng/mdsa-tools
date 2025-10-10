.. _cohesion_density:

Cohesion over time (sliding & shrinking windows)
================================================

Use :class:`mdsa_tools.subdomain_explorations` to quantify **cluster cohesion**
over time in a 2-D embedding (PCA/UMAP). We compute per-cluster RMSD to the
assigned center within moving windows—either **sliding windows** (fixed width,
hop forward) or **shrinking windows** (drop early frames and keep the tail).

What you get
------------
- Windowed **RMSD vs. time** per cluster (as ``.csv`` tables).
- Easy to analyze **line plots** comparing your clusters progressive cohesion score via :func:`mdsa_tools.Viz.rmsd_lineplots`.
- Hooks to extend into subdomain_explorations diagnostics.

Quickstart
----------
Minimal example assuming you already produced 2-D coordinates (``X``) and
K-means labels/centers from :meth:`mdsa_tools.Analysis.systems_analysis.*`.

.. code-block:: python

   import os
   import numpy as np
   import pandas as pd
   import matplotlib.cm as cm

   from mdsa_tools.subdomain_explorations import subdomain_explorations as subdomain_explorations
   from mdsa_tools.Viz import rmsd_lineplots

   #########################################
   # Inputs (embedding + clustering artifacts)
   #########################################
   # X: (n_frames, 2) PCA/UMAP coordinates (concatenated across replicates)
   # labels: (n_frames,) 0-based cluster ids
   # centers: (n_states, 2) cluster centers in the same space
   # frame_scale: list of per-replicate frame counts, in concat order

   X        = np.load("/path/to/embeddings/X_pca_or_umap.npy")          # shape (n_frames, 2)
   labels   = np.load("/path/to/labels/silhouette_opt_labels.npy")      # shape (n_frames,)
   centers  = np.load("/path/to/labels/silhouette_centers.npy")         # shape (n_states, 2)
   frame_scale = [80]*20 + [160]*10                                     # example: 30 reps total

   # (Optional) clean rows with NaNs/Infs
   m = np.isfinite(X).all(axis=1)
   X, labels = X[m], labels[m]

   #########################################
   # Build modeller and run windowed cohesion
   #########################################
   modeller = subdomain_explorations(labels=labels, centers=centers, reduced_coordinates=X, frame_scale=frame_scale)

   # Sliding window: fixed-length chunks that hop forward
   sliding_df   = modeller.evaluate_cohesion_slidingwindow(step_size=20)     # window=20 frames
   shrinking_df = modeller.evaluate_cohesion_shrinkingwindow(step_size=20)   # drop 20 frames each step

   outdir = "/path/to/out/cohesion"
   os.makedirs(outdir, exist_ok=True)

   sliding_csv   = os.path.join(outdir, "rmsd_sliding.csv")
   shrinking_csv = os.path.join(outdir, "rmsd_shrinking.csv")

   sliding_df.to_csv(sliding_csv, index=False, float_format="%.4f")
   shrinking_df.to_csv(shrinking_csv, index=False, float_format="%.4f")

   #########################################
   # Plot RMSD trajectories per cluster
   #########################################
   # The helper expects a DataFrame with columns ['cluster','rmsd','window']
   rmsd_lineplots(sliding_df,   outfilepath=os.path.join(outdir, "rmsd_sliding"),   cmap=cm.plasma)
   rmsd_lineplots(shrinking_df, outfilepath=os.path.join(outdir, "rmsd_shrinking"), cmap=cm.plasma)

   print(f"Saved CSVs:\n  {sliding_csv}\n  {shrinking_csv}")
   print(f"Saved plots to: {outdir}")

Notes
-----
- **Signals:** Lower RMSD implies tighter coherence of frames around each cluster
  center in that window. Look for stabilization (RMSD decreasing) vs. drift.
- **Windows:** ``step_size`` is both the window width (sliding) and the left-edge
  hop (shrinking). Use values that reflect your sampling/τ.
- **Replicates:** ``frame_scale`` prevents cross-replicate jumps; if replicates
  differ in length, short ones naturally drop out of later windows.
- **Sanity checks:** Inspect the CSVs for clusters that vanish in a window
  (they simply won’t appear in that window’s rows).

Where this fits
---------------
- After :ref:`full_example` clustering, run cohesion to assess whether your
  K-means states are kinetically sensible before subdomain_explorations fitting.
- Combine with **replicate maps** (see :mod:`mdsa_tools.Viz`) to visualize
  state usage across replicates in parallel to cohesion trends.

See also
--------
- :class:`mdsa_tools.subdomain_explorations` — cohesion windows, transition matrices,
- :func:`mdsa_tools.Viz.rmsd_lineplots` — render per-cluster RMSD vs. window.
- :meth:`mdsa_tools.Analysis.systems_analysis.perform_kmeans` — produce labels/centers.
- :meth:`mdsa_tools.Analysis.systems_analysis.reduce_systems_representations` — get PCA/UMAP.
