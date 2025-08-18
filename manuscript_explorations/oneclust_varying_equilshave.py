from mdsa_tools.Analysis import systems_analysis
import numpy as np
import matplotlib.cm as cm
import os

#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)

#For the paper we move forward with systems representations
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
Systems_Analyzer = systems_analysis(all_systems)
X_pca,_,_=Systems_Analyzer.reduce_systems_representations(method='PCA',n_components=2) #PCA

from mdsa_tools.Viz import visualize_reduction
global_frame_list=((([80] * 20) + ([160] * 10))*2)
GCUkluster_labels_2clust = np.load('/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/shaved0_shaved_GCUkluster_labels_5clust.npy')
CGUkluster_labels_2clust = np.load('/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/shaved0_shaved_CGUkluster_labels_2clust.npy')



##############################
#just shaving up to the short#
##############################
for i in range(0,60,10):
    #grabbing splices
    X_PCA_cords=[]
    GCU_colors=[]
    CGU_colors=[]
    iterator=0

    for j in global_frame_list:
        spliced_rep=X_pca[iterator+i:iterator+j,:]
        current_GCUkluster_labels_2clust=GCUkluster_labels_2clust[iterator+i:iterator+j]
        current_CGUkluster_labels_2clust=CGUkluster_labels_2clust[iterator+i:iterator+j]
        X_PCA_cords.append(spliced_rep)
        GCU_colors.append(current_GCUkluster_labels_2clust)
        CGU_colors.append(current_CGUkluster_labels_2clust)
        iterator+=j

    X_PCA_cords=np.vstack(X_PCA_cords)
    GCU_colors=np.concatenate(GCU_colors)
    CGU_colors=np.concatenate(CGU_colors)
    GCU_with_filler=np.concatenate((GCU_colors,np.full(shape=(3200-i*30,),fill_value=np.max(GCU_colors)+1)))
    CGU_with_filler=np.concatenate((np.full(shape=(3200-i*30,),fill_value=np.max(CGU_colors)+1),CGU_colors))
    
    visualize_reduction(X_PCA_cords,color_mappings=GCU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/GCU_onlyshaved{i*2}_ns_PCA')#each system
    visualize_reduction(X_PCA_cords,color_mappings=CGU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/CGU_onlyshaved{i*2}_ns_PCA')#each system


#############################
#just shaving up the long#
#############################

XPCA_ids = np.arange(6400) 
XPCA_mask = ((XPCA_ids >= 1600) & (XPCA_ids < 3200)) | ((XPCA_ids >= 4800) & (XPCA_ids < 6400))#bools for mask
colors_ids = np.arange(3200) 
colors_mask = ((colors_ids >= 1600))#bools for mask

GCU_long_labels=GCUkluster_labels_2clust[colors_mask]
CGU_long_labels=CGUkluster_labels_2clust[colors_mask]
X_PCA_long_cords=X_pca[XPCA_mask]
long_frame_list=global_frame_list=(([160] * 10)*2)

for i in range(0,100,10):
    #grabbing splices
    X_PCA_cords=[]
    GCU_colors=[]
    CGU_colors=[]
    iterator=0

    for j in long_frame_list:
        spliced_rep=X_PCA_long_cords[iterator+i:iterator+j,:]
        current_GCUkluster_labels_2clust=GCU_long_labels[iterator+i:iterator+j]
        current_CGUkluster_labels_2clust=CGU_long_labels[iterator+i:iterator+j]
        X_PCA_cords.append(spliced_rep)
        GCU_colors.append(current_GCUkluster_labels_2clust)
        CGU_colors.append(current_CGUkluster_labels_2clust)
        iterator+=j

    X_PCA_cords=np.vstack(X_PCA_cords)
    GCU_colors=np.concatenate(GCU_colors)
    CGU_colors=np.concatenate(CGU_colors)
    GCU_with_filler=np.concatenate((GCU_colors,np.full(shape=(1600-i*10,),fill_value=np.max(GCU_colors)+1)))
    CGU_with_filler=np.concatenate((np.full(shape=(1600-i*10,),fill_value=np.max(CGU_colors)+1),CGU_colors))
    
    visualize_reduction(X_PCA_cords,color_mappings=GCU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/long_GCU_onlyshaved{i*2}_ns_PCA')#each system
    visualize_reduction(X_PCA_cords,color_mappings=CGU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/long_CGU_onlyshaved{i*2}_ns_PCA')#each system
