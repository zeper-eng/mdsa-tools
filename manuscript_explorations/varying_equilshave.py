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

global_frame_list=((([80] * 20) + ([160] * 10))*2)

rmsd_shavings=[]
from mdsa_tools.Viz import visualize_reduction
for i in range(0,70,10):

    #grabbing splices
    X_PCA_cords=[]
    iterator=0
    for j in global_frame_list:
        spliced_rep=X_pca[iterator+i:iterator+j,:]
        X_PCA_cords.append(spliced_rep)
        iterator+=j
    X_PCA_cords=np.vstack(X_PCA_cords)


    frame_list=((([80-i] * 20) + ([160-i] * 10))*2)
    colormappings = np.concatenate([np.arange(j, dtype=float) for j in frame_list])


    
    #Cluster embedding space (PCA)
    optimal_k_silhouette_labels_GCUresults,optimal_k_elbow_labels_GCUresults,centers_sillohuette_GCUresults,centers_elbow_GCUresults=Systems_Analyzer.cluster_system_level(data=X_PCA_cords[0:3200-i*30,:],outfile_path=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/shaved{i*2}_shaved_GCU')
    optimal_k_silhouette_labels_CGUresults,optimal_k_elbow_labels_CGUresults,centers_sillohuette_CGUresults,centers_elbow_CGUresults=Systems_Analyzer.cluster_system_level(data=X_PCA_cords[3200-i*30:,:],outfile_path=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/shaved{i*2}_shaved_CGU')

    #grabbing rmsd
    for k in np.unique(optimal_k_silhouette_labels_GCUresults):
        dists = np.linalg.norm(
            X_PCA_cords[0:3200 - i*30, :][optimal_k_silhouette_labels_GCUresults == k]
            - centers_sillohuette_GCUresults[k],
            axis=1
        )
        rmsd = np.sqrt(np.mean(dists**2))
        rmsd_shavings.append({"shaving": i, "system": "GCU", "cluster": int(k), "RMSD": rmsd})

    for k in np.unique(optimal_k_silhouette_labels_CGUresults):
        dists = np.linalg.norm(
            X_PCA_cords[3200 - i*30:, :][optimal_k_silhouette_labels_CGUresults == k]
            - centers_sillohuette_CGUresults[k],
            axis=1
        )
        rmsd = np.sqrt(np.mean(dists**2))
        rmsd_shavings.append({"shaving": i, "system": "CGU", "cluster": int(k), "RMSD": rmsd})

    GCU_with_filler=np.concatenate((optimal_k_silhouette_labels_GCUresults,np.full(shape=(3200-i*30,),fill_value=np.max(optimal_k_silhouette_labels_GCUresults)+1)))
    CGU_with_filler=np.concatenate((np.full(shape=(3200-i*30,),fill_value=np.max(optimal_k_silhouette_labels_CGUresults)+1),optimal_k_silhouette_labels_CGUresults))


    visualize_reduction(X_PCA_cords,color_mappings=GCU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/GCUshaved{i*2}_ns_PCA')#each system
    visualize_reduction(X_PCA_cords,color_mappings=CGU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/CGUshaved{i*2}_ns_PCA')#each system

names=[f'{i}_shaved' for i in range(0,70,10)]
import pandas as pd
df = pd.DataFrame(rmsd_shavings)
df.to_csv("/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/rmsd_results.txt", sep="\t", index=False)
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


