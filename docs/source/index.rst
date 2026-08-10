
Welcome to MDSA-tools!
======================
.. image:: https://img.shields.io/badge/GitHub-mdsa--tools-f06292?logo=github&logoColor=1E3A8A&labelColor=555555&color=f06292
   :target: https://github.com/zeper-eng/MDSA-tools
   :alt: GitHub: MDSA-tools

.. image:: https://img.shields.io/pypi/v/MDSA-tools?label=PyPI
   :target: https://pypi.org/project/MDSA-tools/
   :alt: PyPI: MDSA-tools

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.17195649.svg
   :target: https://doi.org/10.5281/zenodo.17195649
   :alt: DOI: 10.5281/zenodo.17195649

MDSA-tools is a Python package for systems-level analysis of molecular-dynamics (MD) data. It provides utilities for clustering, dimensionality reduction (PCA, UMAP), visualization, and hydrogen-bond network generation.


.. toctree:: 
   :maxdepth: 2
   :hidden:
   
   GitHub <https://github.com/zeper-eng/MDSA-tools>
   api
   examples

.. grid:: 1 1 1 1
   :gutter: 2
   :margin: 3

   .. grid-item-card:: 
      :link: api
      :shadow: lg
      :link-type: doc       
      :text-align: center 
      
      **Link to the full python API**
      ^^^
      
      .. image:: /resources/pipeline_11_2_2025.png
         :alt: PCA clusters for system A
         :width: 100%
         :align: center

      +++
      **Full description of the modules included as a part of the workflow**


**Excerpt from our paper:**

With the impressive development of force field parameters that allow successful computational simulations of biological molecules, bringing in systems modes of analysis is a natural next step to begin to understand the molecular dynamics behaviors that emerge from these experiments. We think of this as trying out and exploring lenses that can reveal different important behaviors. 

Following the approaches of classical molecular genetics, we use a “computational genetics” paradigm where we introduce changes (mutations) in potentially important residues—changing their identities or modifying their chemical properties—and ask how the dynamic system responds to these changes. 

While some systems analytical approaches are “black box” in nature, making it harder to dissect the basis of observed behaviors, we have also explored network representations that allow us to home in on structural components whose behaviors are altered by the “computational mutations.” Applied to our ribosome neighborhood, this revealed unexpected changes in behavior at the ribosome peptidyl site (P site) in response to mutating mRNA residues next to the aminoacyl site (A site) codon, 
suggesting long-range allosteric interactions across the neighborhood.  

   .. grid-item-card:: 
      :link: https://www.biorxiv.org/content/10.64898/2026.03.28.714829v1
      :link-type: url        
      :text-align: center

      **Link to the Systems Pre-print**
      ^^^

      .. image:: /resources/paperfigure.png
         :alt: Paper figure
         :width: 100%
         :align: center

      +++
      **A Systems Analysis of Ribosomal CAR-site Dynamics**

Focused on leveraging unsupervised learning algorithm workflows our pipeline can build hydrogen-bond adjacency matrices from MDTraj or import them from Cpptraj. By design, the analysis and visualization modules are data-agnostic as long as you fit the expected schema.

   .. grid-item-card:: 
      :link: examples
      :link-type: doc        
      :text-align: center

      **Examples**
      ^^^

      .. image:: /resources/examplescover.png
         :alt: Example cover image
         :width: 100%
         :align: center

      +++
      
      **Use cases from Visualization to Analysis**

 