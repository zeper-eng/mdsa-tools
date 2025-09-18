.. _full_example:


Feature Matrix and exploratory clustering
=========================================

If you separated out the data generation phase and saved your systems earlier, you
can start here. This uses two saved ``.npy`` arrays and runs clustering + reduction
(including an embedding-space clustering pass).

Otherwise see :ref:`datagen`.

.. code-block:: python

   from mdsa_tools.Convenience import unrestrained_residues
   from mdsa_tools.Analysis import systems_analysis
   import os
   import numpy as np

   #We seperate out the data generation phase and assume you generated and saved your data prior
   redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
   redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
   all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]

   Systems_Analyzer = systems_analysis(all_systems)

   #Clustering and visualizing clusters 
   optimal_k_silhouette_labels,optimal_k_elbow_labels,centers_sillohuette,centers_elbow = Systems_Analyzer.cluster_system_level(outfile_path='/Users/luis/Desktop/workspacetwo/manuscript_explorations/syskmeans/',max_clusters=25)
   print('clustering succesfully completed')

   Systems_Analyzer.reduce_systems_representations(outfile_path='/Users/luis/Desktop/workspace/test_output/PCA/test_',colormappings=optimal_k_silhouette_labels) #you could do method=PCA/UMAP here
   print('PCA reduction succesful')

   Systems_Analyzer.cluster_embeddingspace(outfile_path='/Users/luis/Desktop/workspace/test_output/cluster_embeddingspace/',max_clusters=10,elbow_or_sillohuette='sillohuette')
   print('Embedding space clustering succesfully completed')

See also

- :mod:`mdsa_tools.Data_gen_hbond` for creating the per-frame adjacency matrices.
- :class:`mdsa_tools.Analysis.systems_analysis` for clustering and reduction helpers.
- :mod:`mdsa_tools.Viz` for embedding plots and replicate maps.
