from mdsa_tools.Analysis import systems_analysis
import numpy as np
import matplotlib.cm as cm
import os

#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]

#For the paper we move forward with systems representations
Systems_Analyzer = systems_analysis(all_systems)
X_pca,_,_=Systems_Analyzer.reduce_systems_representations(method='PCA',n_components=2,n_neighbors=100,min_dist=0) #PCA


################
#worm behavior#
###############
from mdsa_tools.Viz import visualize_reduction
frame_list=((([80] * 20) + ([160] * 10))*2)
system_labels=(([1]*3200)+[2]*3200)
colormappings=[np.arange(0,np.max(i),1) for i in frame_list]
colormappings=np.concatenate(colormappings)
visualize_reduction(X_pca,color_mappings=system_labels,cmap=cm.magma_r,savepath='/Users/luis/Desktop/workspacetwo/manuscript_explorations/crawlspace/PCA_')#each system
visualize_reduction(X_pca,color_mappings=colormappings,cmap=cm.magma_r,savepath='/Users/luis/Desktop/workspacetwo/manuscript_explorations/crawlspace/PCA_')#all reps

os._exit(0)



#umap below
os._exit(0)
#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]

#For the paper we move forward with systems representations
Systems_Analyzer = systems_analysis(all_systems)
UMAP_coordinates=Systems_Analyzer.reduce_systems_representations(method='UMAP',n_components=2,n_neighbors=100,min_dist=0) #PCA


################
#worm behavior#
###############
from mdsa_tools.Viz import highlight_reps_in_embeddingspace,highlight_crawl_directions,visualize_reduction
frame_list=((([80] * 20) + ([160] * 10))*2)
system_labels=(([1]*3200)+[2]*3200)
colormappings=[np.arange(0,np.max(i),1) for i in frame_list]
colormappings=np.concatenate(colormappings)
visualize_reduction(UMAP_coordinates,color_mappings=system_labels,cmap=cm.magma_r,savepath='/Users/luis/Desktop/workspacetwo/manuscript_explorations/crawlspace/00n_point0mindist')#each system
visualize_reduction(UMAP_coordinates,color_mappings=colormappings,cmap=cm.magma_r,savepath='/Users/luis/Desktop/workspacetwo/manuscript_explorations/crawlspace/00n_point0mindist_crawlspace')#all reps

os._exit(0)