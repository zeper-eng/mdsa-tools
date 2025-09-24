.. _contour_density:

Contours on PCA/UMAP embeddings
===============================

Use :func:`mdsa_tools.Viz.contour_embedding_space` to render a **density contour map**
over 2-D embedding coordinates (from PCA or UMAP). This is handy for spotting
high-density basins and comparing to your K-means labels when identifying potential
candidate states.

What you get
------------
- A high-resolution ``.png`` (or any Matplotlib-supported format) with filled contours.
- Optional title/axis labels and grid toggles.
- Simple knobs for contour detail (``levels``) and smoothness (``bw_adjust``).

Quickstart
----------
Minimal example assuming you already produced a 2-D embedding with
:meth:`mdsa_tools.Analysis.systems_analysis.reduce_systems_representations` and saved
the coordinates to ``.npy`` (shape ``(n_samples, 2)``).

.. code-block:: python

   import os
   import numpy as np
   from mdsa_tools.Viz import contour_embedding_space

   #########################################
   # Load 2-D embedding (PCA or UMAP)
   #########################################
   # Example: file saved earlier during reduction; shape (n_samples, 2)
   emb_path = "/path/to/output/reduction/test_embedding_coords.npy"
   E = np.load(emb_path)


   #########################################
   # Render density contours
   #########################################
   out_img = "/path/to/output/reduction/pca_density_contours.png"
   contour_embedding_space(
       outfile_path=out_img,
       embeddingspace_coordinates=E,
       levels=12,            # more/less contour bands
       thresh=0.0,           # >0 to clip ultra-low-density fringes
       bw_adjust=0.6,        # larger -> smoother; smaller -> sharper
       title="PCA density (PC1 vs PC2)",
       xlabel="PC1",
       ylabel="PC2",
       gridvisible=False
   )
   print(f"Saved: {out_img}")

Notes
-----
- **Input shape:** pass a ``(n_samples, 2)`` array (columns = the two embedding axes).
- **Smoothing:** ``bw_adjust`` scales Seaborn’s KDE bandwidth. Values ``0.4–0.8`` are
  a good starting range; increase if your plot looks choppy.
- **Clipping:** raise ``thresh`` (e.g., ``0.05``) to suppress faint outer rings.
- **File type:** the extension of ``outfile_path`` controls the format (``.png``,
  ``.pdf``, ``.svg``, …).
- **Pair with clusters:** plot this alongside your label-colored scatter to check that
  clusters align with density basins (see :ref:`full_example` for label generation).

See also
--------
- :meth:`mdsa_tools.Analysis.systems_analysis.reduce_systems_representations` — produce 2-D PCA/UMAP.
- :meth:`mdsa_tools.Analysis.systems_analysis.cluster_embeddingspace` — label points directly in embedding space.
- :mod:`mdsa_tools.Viz` — additional plotting helpers.