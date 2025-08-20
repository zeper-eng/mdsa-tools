from mdsa_tools.Analysis import systems_analysis
import numpy as np
import matplotlib.cm as cm
import os
import pandas as pd


#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)

from mdsa_tools.Viz import visualize_reduction
persys_frame_list=((([80] * 20) + ([160] * 10)))

#For the paper we move forward with systems representations
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
Systems_Analyzer = systems_analysis(systems_representations=all_systems,replicate_distribution=persys_frame_list)
Systems_Analyzer.replicates_to_featurematrix()
X_pca,_ ,_=Systems_Analyzer.reduce_systems_representations()
candidate_states_per_system=Systems_Analyzer.cluster_embeddingspace()

#visualize candidate states
from mdsa_tools.Viz import visualize_candidate_states
visualize_candidate_states(candidate_states_per_system,X_pca,cmap=cm.inferno_r)
from mdsa_tools.msm_modeler import MSM_Modeller as msm

MSM_modeler=msm(candidate_states_per_system,X_pca,persys_frame_list)
RMSD_dataframe=MSM_modeler.evaluate_cohesion(window=20)
RMSD_dataframe.to_csv('RMSD_dataframe.csv')
X_pca,_,_=Systems_Analyzer.reduce_systems_representations(method='PCA',n_components=2) #PCA



os._exit(0)

###
###Unique Case
###
optimal_k_silhouette_labels_GCUresults,optimal_k_elbow_labels_GCUresults,centers_sillohuette_GCUresults,centers_elbow_GCUresults=Systems_Analyzer.cluster_system_level(data=X_pca[0:3200,:],outfile_path='/Users/luis/Desktop/workspacetwo/manuscript_explorations//embeddingspace_kmeanslabels/GCU')
optimal_k_silhouette_labels_CGUresults,optimal_k_elbow_labels_CGUresults,centers_sillohuette_CGUresults,centers_elbow_CGUresults=Systems_Analyzer.cluster_system_level(data=X_pca[3200:,:],outfile_path='/Users/luis/Desktop/workspacetwo/manuscript_explorations//embeddingspace_kmeanslabels/CGU')

rmsd_shavings_all=[]
slice_names=[]

for i in range(0,60,10):
    mask=build_shave_mask(persys_frame_list,i)
    onesysmask=build_shave_mask(onesys_frame_list,i)
    #building our much needed masks
    masked_coordinates=X_pca[mask,:]

    #grabbing only what were interested in and adding in some filler labels
    masked_PCA_labels_GCU,masked_PCA_labels_CGU=optimal_k_silhouette_labels_GCUresults[onesysmask],optimal_k_silhouette_labels_CGUresults[onesysmask]
    GCU_with_filler=np.concatenate((masked_PCA_labels_GCU,np.full(shape=(3200-i*30,),fill_value=np.max(masked_PCA_labels_GCU)+1)))
    CGU_with_filler=np.concatenate((np.full(shape=(3200-i*30,),fill_value=np.max(masked_PCA_labels_CGU)+1),masked_PCA_labels_CGU))
    
    #visualize
    visualize_reduction(masked_coordinates,color_mappings=GCU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/all_GCU_onlyshaved{i*2}_ns_PCA')#each system
    visualize_reduction(masked_coordinates,color_mappings=CGU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/all_CGU_onlyshaved{i*2}_ns_PCA')#each system

    rmsd_results = rmsd_from_centers(
        masked_coordinates[:masked_PCA_labels_GCU.shape[0]],
        masked_PCA_labels_GCU,
        centers_sillohuette_GCUresults
    )
    df_temp = pd.DataFrame(rmsd_results, columns=["cluster", "RMSD"])
    df_temp["slice"] = f"shaved_{i*2}ns"
    rmsd_shavings_all.append(df_temp)

# concat everything together
rmsd_df = pd.concat(rmsd_shavings_all, ignore_index=True)
rmsd_df.to_csv("/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/rmsd_results.txt", sep="\t", index=False)
print(rmsd_df)



#####################################################################################################################################################################
#######################ONLY LONG#####################################################################################################################################
#####################################################################################################################################################################


############################
#functions for this process#
############################

long_idx = np.r_[1600:3200, 4800:6400] #i did not know this existed!concaenated index array!
X_long = X_pca[long_idx, :] 

persys_frame_list=(( ([160] * 10))*2)
onesys_frame_list=(( ([160] * 10)))
long_optimal_k_silhouette_labels_GCUresults,long_optimal_k_silhouette_labels_CGUresults=optimal_k_silhouette_labels_GCUresults[1600:3200] ,optimal_k_silhouette_labels_CGUresults[1600:3200] 

rmsd_shavings_all=[]
slice_names=[]

for i in range(0,100,10):
    mask=build_shave_mask(persys_frame_list,i)
    onesysmask=build_shave_mask(onesys_frame_list,i)
    #building our much needed masks
    masked_coordinates=X_long[mask,:]

    #grabbing only what were interested in and adding in some filler labels
    masked_PCA_labels_GCU,masked_PCA_labels_CGU=long_optimal_k_silhouette_labels_GCUresults[onesysmask],long_optimal_k_silhouette_labels_CGUresults[onesysmask]
    GCU_with_filler=np.concatenate((masked_PCA_labels_GCU,np.full(shape=masked_PCA_labels_GCU.shape[0],fill_value=np.max(masked_PCA_labels_GCU)+1)))
    CGU_with_filler=np.concatenate((np.full(shape=masked_PCA_labels_GCU.shape[0],fill_value=np.max(masked_PCA_labels_CGU)+1),masked_PCA_labels_CGU))
    
    #visualize
    visualize_reduction(masked_coordinates,color_mappings=GCU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/long_GCU_onlyshaved{i*2}_ns_PCA')#each system
    visualize_reduction(masked_coordinates,color_mappings=CGU_with_filler,cmap=cm.magma,savepath=f'/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/long_CGU_onlyshaved{i*2}_ns_PCA')#each system

    rmsd_results = rmsd_from_centers(
        masked_coordinates[:masked_PCA_labels_GCU.shape[0]],
        masked_PCA_labels_GCU,
        centers_sillohuette_GCUresults
    )
    
    df_temp = pd.DataFrame(rmsd_results, columns=["cluster", "RMSD"])
    df_temp["slice"] = f"shaved_{i*2}ns"
    rmsd_shavings_all.append(df_temp)

# concat everything together
rmsd_df = pd.concat(rmsd_shavings_all, ignore_index=True)
rmsd_df.to_csv("/Users/luis/Desktop/workspacetwo/manuscript_explorations/shaved_pca_ns/long_rmsd_results.txt", sep="\t", index=False)
print(rmsd_df)
os._exit(0)