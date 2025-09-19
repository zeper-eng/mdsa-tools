.. _cpptraj_hbond_import:

cpptraj (series) → residue×residue H-bond matrices
==================================================

Use :class:`mdsa_tools.Cpptraj_import.cpptraj_hbond_import` to parse a cpptraj
``hbond ... out <file> series`` table into residue–residue time-series matrices
compatible with the rest of the pipeline.

What you get
------------
- Parsed residue index pairs in column order (``loader.indices``).
- The raw series matrix of shape ``(n_frames, n_pairs)`` (``loader.data``).
- A stack of per-frame residue×residue adjacency matrices
  ``(n_frames, n_res+1, n_res+1)`` via :meth:`~mdsa_tools.Cpptraj_import.cpptraj_hbond_import.create_systems_rep`.

Quickstart
----------
Minimal example using a topology readable by MDTraj (e.g., AMBER ``.prmtop``) and a
cpptraj ``series`` table. If you’re unfamiliar with generating these, see :ref:`datagen`.

**1) Create the `series` table with cpptraj**

.. code-block:: bash

   parm system.prmtop
   trajin traj.nc
   hbond HB out hbonds.dat series
   run
   quit

Run with:

.. code-block:: bash

   cpptraj -i cpptraj.in

**2) Load and build matrices**

.. code-block:: python

   from mdsa_tools.Cpptraj_import import cpptraj_hbond_import
   import numpy as np
   import os

   series_path = "/path/to/hbonds.dat"
   topology_path = "/path/to/system.prmtop"

   loader = cpptraj_hbond_import(series_path, topology_path)
   print("pairs:", loader.indices[:5])
   print("series shape:", loader.data.shape)  # (n_frames, n_pairs)

   systems = loader.create_systems_rep()      # (n_frames, n_res+1, n_res+1)
   np.save(os.path.join("/path/to/out", "hbonds_residue_series.npy"), systems)

Notes
-----
- Matrices store **1-based** residue labels in row/column ``0``; numeric adjacency is ``[1:, 1:]``.
- Self-contacts are zero; no symmetrization is applied by default.

Where this fits
---------------
- Generate adjacency stacks here, then pass them to
  :class:`mdsa_tools.Analysis.systems_analysis` or plot with :mod:`mdsa_tools.Viz`.

See also
--------
- :mod:`mdsa_tools.Data_gen_hbond` — build per-frame H-bond matrices from trajectories.
- :mod:`mdsa_tools.Cpptraj_import` — parse cpptraj ``series`` tables.
- :class:`mdsa_tools.Analysis.systems_analysis` — downstream feature building and PCA.
