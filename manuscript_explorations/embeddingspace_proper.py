from utilities.Analysis import systems_analysis
import numpy as np
import matplotlib.cm as cm

#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]

#For the paper we move forward with systems representations
Systems_Analyzer = systems_analysis(all_systems)
X_pca,_,_=Systems_Analyzer.reduce_systems_representations(method='PCA',n_components=2) #PCA

#Contour embedding space 
from utilities.Viz import contour_embedding_space
contour_embedding_space('/Users/luis/Desktop/workspacetwo/manuscript_explorations/contour/contour_test_PCA',X_pca)#PCA

#Cluster embedding space (PCA)
optimal_k_silhouette_labels_GCUresults,optimal_k_elbow_labels_GCUresults,centers_sillohuette_GCUresults,centers_elbow_GCUresults=Systems_Analyzer.cluster_system_level(data=X_pca[0:3200,:],outfile_path='/Users/luis/Desktop/workspacetwo/manuscript_explorations//embeddingspace_kmeanslabels/GCU')
optimal_k_silhouette_labels_CGUresults,optimal_k_elbow_labels_CGUresults,centers_sillohuette_CGUresults,centers_elbow_CGUresults=Systems_Analyzer.cluster_system_level(data=X_pca[3200:,:],outfile_path='/Users/luis/Desktop/workspacetwo/manuscript_explorations//embeddingspace_kmeanslabels/CGU')
print(optimal_k_silhouette_labels_GCUresults.shape)
print(optimal_k_silhouette_labels_CGUresults.shape)

#################################################################################################
## We will need the breakdown of which frames belong to which set of trajectories moving forward#
# for the first appication we only use one system at a time so we keep it like this but then we #
# use two versions in the next implementation.                                                  #
#################################################################################################

frame_list=((([80] * 20) + ([160] * 10)))
print(len(frame_list))
print(sum(frame_list))

#############################
#rest of worklow begins here#
#############################

#Create replicate maps of embeddingspace results
from utilities.Viz import replicatemap_from_labels
replicatemap_from_labels(optimal_k_silhouette_labels_GCUresults,frame_list=frame_list,savepath='/Users/luis/Desktop/workspacetwo/manuscript_explorations/replicate_maps/GCU_embeddingspace_',title='GCU_substate_replicatemap')
replicatemap_from_labels(optimal_k_silhouette_labels_CGUresults,frame_list=frame_list,savepath='/Users/luis/Desktop/workspacetwo/manuscript_explorations/replicate_maps/CGU_embeddingspace_',title='CGU_substate_replicatemap')

#Visualize embeddingspace results on embeddingspace 
GCU_with_filler=np.concatenate((optimal_k_silhouette_labels_GCUresults,np.full(shape=(3200,),fill_value=10)))
CGU_with_filler=np.concatenate((np.full(shape=(3200,),fill_value=10),optimal_k_silhouette_labels_CGUresults))

from utilities.Viz import visualize_reduction
visualize_reduction(X_pca,color_mappings=CGU_with_filler,cmap=cm.plasma,savepath='/Users/luis/Desktop/workspacetwo/manuscript_explorations/embeddingspace_visualizations/CGU_embeddingspacecluster_visualizations')
visualize_reduction(X_pca,color_mappings=GCU_with_filler,cmap=cm.plasma,savepath='/Users/luis/Desktop/workspacetwo/manuscript_explorations/embeddingspace_visualizations/GCU_embeddingspacecluster_visualizations')

from utilities.Analysis import MSM_Modeller

frame_list*=2 #updating frame_list here
MSM_GCU=MSM_Modeller(optimal_k_silhouette_labels_GCUresults,frame_list)
MSM_CGU=MSM_Modeller(optimal_k_silhouette_labels_CGUresults,frame_list)

transtion_prob_matrix_GCU = MSM_GCU.create_transition_probability_matrix()
transtion_prob_matrix_CGU = MSM_CGU.create_transition_probability_matrix()

print(transtion_prob_matrix_GCU)
print(transtion_prob_matrix_CGU)

np.save('/Users/luis/Desktop/workspacetwo/manuscript_explorations/transition_probability_matrices/transtion_prob_matrix_GCU',transtion_prob_matrix_GCU)
np.save('/Users/luis/Desktop/workspacetwo/manuscript_explorations/transition_probability_matrices/transtion_prob_matrix_CGU',transtion_prob_matrix_CGU)

            