from mdsa_tools.Analysis import systems_analysis
import numpy as np
import matplotlib.cm as cm
import os
import pandas as pd

#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)

#For the paper we move forward with systems representations
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
Systems_Analyzer = systems_analysis(all_systems)
X_pca,_,_=Systems_Analyzer.reduce_systems_representations(method='PCA',n_components=2) #PCA
optimal_k_silhouette_labels_GCUresults,optimal_k_elbow_labels_GCUresults,centers_sillohuette_GCUresults,centers_elbow_GCUresults=Systems_Analyzer.cluster_system_level(data=X_pca[0:3200,:],outfile_path='/Users/luis/Desktop/workspacetwo/manuscript_explorations//embeddingspace_kmeanslabels/GCU')
optimal_k_silhouette_labels_CGUresults,optimal_k_elbow_labels_CGUresults,centers_sillohuette_CGUresults,centers_elbow_CGUresults=Systems_Analyzer.cluster_system_level(data=X_pca[3200:,:],outfile_path='/Users/luis/Desktop/workspacetwo/manuscript_explorations//embeddingspace_kmeanslabels/CGU')

from mdsa_tools.Viz import visualize_reduction
global_frame_list=((([80] * 20) + ([160] * 10))*2)
GCUkluster_labels_2clust = np.load('/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/shaved0_shaved_GCUkluster_labels_5clust.npy')
CGUkluster_labels_2clust = np.load('/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/shaved0_shaved_CGUkluster_labels_2clust.npy')


def build_shave_mask(frame_list,skip):
    mask=np.concatenate([np.arange(L) >= skip for L in frame_list])#build true false mask based on initial skips (bc arange goes 1,2,3,4 etc)
    return mask

def rmsd_from_centers(X,labels,centers):
    results=[]
    for k in np.unique(labels):
        m = (labels == k) #using yet another mask to filter for 
        
        distances = np.linalg.norm(   
                X[m] - centers[k],
                axis=1
            )
        results.append((          
                int(k),
                float(np.sqrt(np.mean(distances**2)))
            ))
    return results

rmsd_shavings_all=[]

for i in range(0,60,10):
    #grabbing splices
    
    X_PCA_cords=[]
    GCU_colors=[]
    CGU_colors=[]
    sliced_optimal_k_silhouette_centers_GCUresults=[]
    sliced_optimal_k_silhouette_centers_CGUresults=[]
    iterator=0
    
##############################
#just shaving up to the short#
##############################
rmsd_shavings_all=[]

for i in range(0,60,10):
    #grabbing splices

    X_PCA_cords=[]
    GCU_colors=[]
    CGU_colors=[]
    sliced_optimal_k_silhouette_centers_GCUresults=[]
    sliced_optimal_k_silhouette_centers_CGUresults=[]
    iterator=0

    for j in global_frame_list:
        spliced_rep=X_pca[iterator+i:iterator+j,:]
        current_GCUkluster_labels_2clust=GCUkluster_labels_2clust[iterator+i:iterator+j]
        current_CGUkluster_labels_2clust=CGUkluster_labels_2clust[iterator+i:iterator+j]
        current_centers_sillohuette_GCUresults=centers_sillohuette_GCUresults[iterator+i:iterator+j]
        current_centers_sillohuette_CGUresults=centers_sillohuette_CGUresults[iterator+i:iterator+j]
        X_PCA_cords.append(spliced_rep)
        GCU_colors.append(current_GCUkluster_labels_2clust)
        CGU_colors.append(current_CGUkluster_labels_2clust)
        sliced_optimal_k_silhouette_centersGCUresults.append(current_centers_sillohuette_GCUresults)
        sliced_optimal_k_silhouette_centers_CGUresults.append(current_centers_sillohuette_CGUresults)
        iterator+=j

    X_PCA_cords=np.vstack(X_PCA_cords)
    GCU_colors=np.concatenate(GCU_colors)
    CGU_colors=np.concatenate(CGU_colors)
    GCU_with_filler=np.concatenate((GCU_colors,np.full(shape=(3200-i*30,),fill_value=np.max(GCU_colors)+1)))
    CGU_with_filler=np.concatenate((np.full(shape=(3200-i*30,),fill_value=np.max(CGU_colors)+1),CGU_colors))
    sliced_optimal_k_silhouette_centersGCUresults=np.concatenate(sliced_optimal_k_silhouette_centers_GCUresults)
    sliced_optimal_k_silhouette_centers_CGUresults=np.concatenate(sliced_optimal_k_silhouette_centers_CGUresults)

    for k in np.unique(sliced_optimal_k_silhouette_centers_GCUresults):
        dists = np.linalg.norm(
            X_PCA_cords[0:3200 - i*30, :][sliced_optimal_k_silhouette_centersGCUresults == k]
            - sliced_optimal_k_silhouette_centersGCUresults[k],
            axis=1
        )
        rmsd = np.sqrt(np.mean(dists**2))
        rmsd_shavings_all.append({"shaving": i, "system": "GCU", "cluster": int(k), "RMSD": rmsd})

    for k in np.unique(sliced_optimal_k_silhouette_centers_CGUresults):
        dists = np.linalg.norm(
            X_PCA_cords[3200 - i*30:, :][sliced_optimal_k_silhouette_centers_CGUresults == k]
            - sliced_optimal_k_silhouette_centers_CGUresults[k],
            axis=1
        )
        rmsd = np.sqrt(np.mean(dists**2))
        rmsd_shavings_all.append({"shaving": i, "system": "CGU", "cluster": int(k), "RMSD": rmsd})

    visualize_reduction(X_PCA_cords,color_mappings=GCU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/GCU_onlyshaved{i*2}_ns_PCA')#each system
    visualize_reduction(X_PCA_cords,color_mappings=CGU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/CGU_onlyshaved{i*2}_ns_PCA')#each system

names=[f'{i}_shaved' for i in range(0,60,10)]
df = pd.DataFrame(rmsd_shavings_all)
df.to_csv("/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/rmsd_results.txt", sep="\t", index=False)


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
long_centers=centers_sillohuette_GCUresults[colors_mask]
long_centers=centers_sillohuette_CGUresults[colors_mask]
long_frame_list=global_frame_list=(([160] * 10)*2)

rmsd_shavings_long=[]
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
    
    # GCU long
    for k in np.unique(GCU_colors):
        dists = np.linalg.norm(
            X_PCA_cords[GCU_colors == k] - centers_sillohuette_GCUresults[k],
            axis=1
        )
        rmsd = np.sqrt(np.mean(dists**2))
        rmsd_shavings_long.append({"shaving": i, "system": "GCU-long", "cluster": int(k), "RMSD": rmsd})

    # CGU long
    for k in np.unique(CGU_colors):
        dists = np.linalg.norm(
            X_PCA_cords[CGU_colors == k] - centers_sillohuette_CGUresults[k],
            axis=1
        )
        rmsd = np.sqrt(np.mean(dists**2))
        rmsd_shavings_long.append({"shaving": i, "system": "CGU-long", "cluster": int(k), "RMSD": rmsd})

    visualize_reduction(X_PCA_cords,color_mappings=GCU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/long_GCU_onlyshaved{i*2}_ns_PCA')#each system
    visualize_reduction(X_PCA_cords,color_mappings=CGU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/long_CGU_onlyshaved{i*2}_ns_PCA')#each system

names=[f'{i}_shaved' for i in range(0,100,10)]
df = pd.DataFrame(rmsd_shavings_long)
df.to_csv("/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/rmsd_results_long.txt", sep="\t", index=False)
