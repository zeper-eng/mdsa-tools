
Welcome to mdsa-tools!
======================


mdsa-tools is a Python package for systems-level analysis of molecular-dynamics (MD) data. It provides utilities for clustering, dimensionality reduction (PCA, UMAP), visualization, and hydrogen-bond network generation.

Focused on leveraging unsupervised learning algorithm workflows—PCA/UMAP, clustering, and Markov state modeling (MSM)—it can build hydrogen-bond adjacency matrices from MDTraj or import them from Cpptraj.
By design, the analysis and visualization modules are data-agnostic: they operate on either (i) an n×n adjacency matrix or (ii) an edge list (long-format table) of (i, j, weight) in the expected schema.


.. toctree:: 
   :maxdepth: 1
   :hidden:
   
   api
   examples


.. 2 columns no matter what kind of screen and spacing of 2
.. grid:: 3 3 3 3
   :gutter: 3

   .. grid-item-card:: 
      :link: api
      :shadow: lg
      :link-type: doc       
      :text-align: center
      
      **Link to the full python API**
      ^^^
      
      .. image:: /resources/Pipelineflic.png
         :alt: PCA clusters for system A
         :width: 100%
         :align: center


      +++
      **Full description of the modules included as a part of the workflow**

   
   .. grid-item-card:: 
      :link: examples
      :link-type: doc        
      :text-align: center

      **Examples**
      
      ^^^
      .. image:: /resources/examplescover.png
         :alt: PCA clusters for system A
         :width: 100%
         :align: center


      +++
      **Examples of doing all range of things from visualizations to analysis**


   .. grid-item-card:: 
      :link: api
      :link-type: doc        
      :text-align: center

      **link to the Systems Paper**

      ^^^

      .. image:: /resources/paperfigure.png
         :alt: PCA clusters for system A
         :width: 100%
         :align: center

      +++
      **A Systems Analysis of Ribosomal CAR-site Dynamics**
      
   

