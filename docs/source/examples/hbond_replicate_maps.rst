.. _replicatemap:

Replicate maps (discrete H-bond labels)
=======================================

Use :class:`mdsa_tools.Analysis.systems_analysis` to bind previously saved H-bond adjacency arrays, then pull per-frame, discrete H-bond **labels** for specific residue pairs. Finally, plot replicate maps with :func:`mdsa_tools.Viz.replicatemap_from_labels` to compare label dynamics across frames/replicates.

What you get
------------

- A :class:`~mdsa_tools.Analysis.systems_analysis` instance bound to your systems.
- Per-frame integer labels (or counts/states) for the requested residue pairs.
- Replicate map images saved to disk—useful for quick visual comparisons across replicates and time windows.

Quickstart
----------

Minimal example using two pre-generated systems (``.npy`` arrays produced by :class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor` or :mod:`mdsa_tools.Cpptraj_import`).  
If you’re unfamiliar with producing these arrays, see :ref:`datagen`.

.. code-block:: python

   import os
   import numpy as np

   from mdsa_tools.Analysis import systems_analysis
   from mdsa_tools.Viz import replicatemap_from_labels

   #########################################
   # Load systems (each is a list of frames)
   #########################################
   # Each system should be an array-like of shape:
   # (n_frames, n_res+1, n_res+1), where the leading row/col store 1-based residue IDs.
   redone_CCU_GCU_fulltraj = np.load(
       "/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy",
       allow_pickle=True
   )
   redone_CCU_CGU_fulltraj = np.load(
       "/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy",
       allow_pickle=True
   )
   all_systems = [redone_CCU_GCU_fulltraj, redone_CCU_CGU_fulltraj]

   #########################################
   # Build analyzer and compute features
   #########################################
   Systems_Analyzer = systems_analysis(all_systems)
   # Creates a feature matrix for downstream extraction/label ops
   Systems_Analyzer.replicates_to_featurematrix()

   #########################################
   # Extract discrete labels for residue pairs
   #########################################
   # Example pairs: (411, 422) and (412, 422)
   hbond_counts_411_422 = Systems_Analyzer.extract_hbond_values(residues=[411, 422])
   hbond_counts_412_422 = Systems_Analyzer.extract_hbond_values(residues=[412, 422])

   #########################################
   # Make a frame window list (for map layout)
   #########################################
   # Here we build two blocks of frames per replicate: 20*80 and 10*160,
   # and then duplicate for two systems/replicates -> total length matches labels.
   frames = (([80] * 20) + ([160] * 10)) * 2

   #########################################
   # Plot replicate maps
   #########################################
   outdir = "/Users/luis/Downloads/replicate_maps"
   os.makedirs(outdir, exist_ok=True)

   replicatemap_from_labels(
       labels=hbond_counts_411_422,
       frame_list=frames,
       savepath=os.path.join(outdir, "replicatemap_discrete_411_422")
   )

   replicatemap_from_labels(
       labels=hbond_counts_412_422,
       frame_list=frames,
       savepath=os.path.join(outdir, "replicatemap_discrete_412_422")
   )

Notes
-----

- **Inputs:** Each system is a per-frame H-bond adjacency stack of shape ``(n_frames, n_res+1, n_res+1)``, with the 0-th row/col storing **1-based** residue indices. Slice ``[:, 1:, 1:]`` if you only need the numeric adjacency.
- **Feature prep:** Call :meth:`~mdsa_tools.Analysis.systems_analysis.replicates_to_featurematrix` before extracting labels so that downstream methods have aligned internal representations.
- **Labels shape:** ``labels`` returned by :meth:`~mdsa_tools.Analysis.systems_analysis.extract_hbond_values` should align with your total frames across replicates/systems. Your ``frame_list`` must be the same total length.
- **Windowing:** The ``frame_list`` can segment the visualization by blocks (e.g., different sampling intervals or concatenated runs). Adjust the counts to match your actual concatenation scheme.
- **Saving:** :func:`~mdsa_tools.Viz.replicatemap_from_labels` writes figures to ``savepath`` (without extension); ensure the directory exists.

Where this fits
---------------

- Generate the adjacency arrays with :class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor` (or build from cpptraj via :mod:`mdsa_tools.Cpptraj_import`), then pass them here.
- After inspecting replicate maps, you can:
  - Perform dimensionality reduction or clustering with :mod:`mdsa_tools.Analysis`.
  - Visualize embeddings or MDcircos contact weights with :mod:`mdsa_tools.Viz`.
  - Feed features into subdomain workflows (see :mod:`mdsa_tools.subdomain_explorations`).

See also
--------

- :mod:`mdsa_tools.Data_gen_hbond` — build per-frame H-bond adjacency matrices.
- :mod:`mdsa_tools.Cpptraj_import` — construct the same matrices from cpptraj series tables.
- :meth:`mdsa_tools.Analysis.systems_analysis.extract_hbond_values` — pull per-frame labels/counts for residue pairs.
- :func:`mdsa_tools.Viz.replicatemap_from_labels` — render replicate maps from discrete label sequences.
