
Welcome to mdsa-tools!
======================


mdsa-tools is a python package for systems-level analyses of molecular dynamics data.
It provides tools for clustering, dimensionality reduction, visualization,
and hydrogen bond network generation.

Focused on classical unsupervised methods (PCA/UMAP, clustering, MSM), mdsa-tools can generate residue–residue 
adjacency matrices from mdtraj or import them from cpptraj. 

A fortunate side-effect is that the analysis and viz modules are data-agnostic: any dataset matching the adjacency-matrix schema will work. 

.. image:: ../resources/Pipelineflic.png
   :alt: PCA clusters for system A
   :width: 60%
   :align: center

.. 2 columns no matter what kind of screen and spacing of 2
.. grid:: 2 2 2 2
   :gutter: 2

   .. grid-item-card:: Link to the full python API
      :link: api
      :link-type: doc        
      :text-align: center

   
   .. grid-item-card:: Link to the systems paper
      :link: api
      :link-type: doc        
      :text-align: center

      The paper as the basis for aggregation into modules

.. toctree::
   :hidden:

   api