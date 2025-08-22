#####
#UMAP 

from mdsa_tools.Analysis import systems_analysis
import numpy as np
import matplotlib.cm as cm
import os
import pandas as pd
from mdsa_tools.msm_modeler import MSM_Modeller as msm


#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)

from mdsa_tools.Viz import visualize_reduction
persys_frame_list=((([80] * 20) + ([160] * 10)))
persys_frame_short=([80] * 20) 
persys_frame_long= ([160] * 10)

#For the paper we move forward with systems representations
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
Systems_Analyzer = systems_analysis(systems_representations=all_systems,replicate_distribution=persys_frame_list)
Systems_Analyzer.replicates_to_featurematrix()
X_pca,_ ,_=Systems_Analyzer.reduce_systems_representations()
candidate_states_per_system=Systems_Analyzer.cluster_embeddingspace()

#seperate umap visualization
full_labels_GCU = np.concatenate([
    candidate_states_per_system[0][0],
    np.full(len(candidate_states_per_system[0][0]), np.max(candidate_states_per_system[0][0]) + 1),
])


full_labels_CGU = np.concatenate([
    np.full(len(candidate_states_per_system[1][0]), np.max(candidate_states_per_system[1][0]) + 1),
     candidate_states_per_system[1][0]
])


#UMAP 200 neighbors .3 distance
umap_embedding=Systems_Analyzer.reduce_systems_representations(method='UMAP',n_neighbors=200,min_dist=.3)

visualize_reduction(umap_embedding,full_labels_GCU,'embeddingspace clusters GCU 200 neighbors .3 min dist',cmap=cm.magma_r,savepath='./embeddingspace_visualizations/point3mindist_200_embeddingspace_clusters_GCU')
visualize_reduction(umap_embedding,full_labels_CGU,'embeddingspace clusters CGU 200 neighbors .3 min dist',cmap=cm.magma_r,savepath='./embeddingspace_visualizations/point3mindist_200_embeddingspace_clusters_CGU')


#UMAP 100 neighbors .3 distance
umap_embedding=Systems_Analyzer.reduce_systems_representations(method='UMAP',n_neighbors=100,min_dist=.3)

visualize_reduction(umap_embedding,full_labels_GCU,'embeddingspace clusters GCU 100 neighbors .3 min dist',cmap=cm.magma_r,savepath='./embeddingspace_visualizations/point3mindist_100_embeddingspace_clusters_GCU')
visualize_reduction(umap_embedding,full_labels_CGU,'embeddingspace clusters CGU 100 neighbors .3 min dist',cmap=cm.magma_r,savepath='./embeddingspace_visualizations/point3mindist_100_embeddingspace_clusters_CGU')



from mdsa_tools.Viz import highlight_reps_in_embeddingspace,highlight_crawl_directions,visualize_reduction
frame_list=((([80] * 20) + ([160] * 10))*2)
system_labels=(([1]*3200)+[2]*3200)
colormappings=[np.arange(0,np.max(i),1) for i in frame_list]
colormappings=np.concatenate(colormappings)

visualize_reduction(umap_embedding,color_mappings=system_labels,cmap=cm.magma_r,savepath='/Users/luis/Desktop/workspacetwo/manuscript_explorations/crawlspace/15n_point3mindist',title='UMAP Dimensional Reduction with per-replicate frame highlighting',cbar_label='System')#each system
visualize_reduction(umap_embedding,color_mappings=colormappings,cmap=cm.magma_r,savepath='/Users/luis/Desktop/workspacetwo/manuscript_explorations/crawlspace/15n_point3mindist_crawlspace',title='UMAP Dimensional Reduction with per-replicate frame highlighting',cbar_label='Frame Number')#all reps
