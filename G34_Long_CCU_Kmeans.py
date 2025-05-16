import numpy as np
import sys
sys.path.append("/zfshomes/lperez/final_thesis_scripts/pypure/utilities")
from t_a_Manipulation import replicates_to_featurematrix
from Clustering import preform_clust,preform_clust_opt
from Viz import label_iterator,traj_view_replicates_10by10


#------------------------------------------
#Loading in our files of interest and creating feature matrix
#------------------------------------------

#Load in arrays and make them a list then concatenate them
CCU_GCU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/long_CCU_GCU_Trajectory_array.npy',allow_pickle=True) # all 60 replicates
arrays=[CCU_GCU_fulltraj]
feature_matrix = replicates_to_featurematrix(arrays=arrays)
print(feature_matrix.shape)




#--------------------------------------------------------------------------------------------
#Calculating k-10 with optimization (shown first up next you can manually create k=2-n plots)
#--------------------------------------------------------------------------------------------


replicate_frames=((([80]*20)+([160]*10))*2)
max_clusters=10

optimal_k_silhouette_labels,optimal_k_elbow_labels=preform_clust_opt(feature_matrix,"/zfshomes/lperez/thesis_figures/kmeans/long_",10)
print(optimal_k_silhouette_labels.shape,optimal_k_elbow_labels.shape)

silhouette_viz_df=label_iterator(optimal_k_silhouette_labels,frame_list=replicate_frames)
optimal_k_elbow_labels = label_iterator(optimal_k_elbow_labels,frame_list=replicate_frames)

traj_view_replicates_10by10(silhouette_viz_df,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/long_optimal_silhouette_clust.png",clustering=False)
traj_view_replicates_10by10(optimal_k_elbow_labels,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/long_optimal_optimal_elbow_clust.png",clustering=False)




import os
os._exit(0)
#------------------------------------------
#Calculating k-10 without optimization
#------------------------------------------

# This section is just automation of the creation of our plots

#example case of frame number list
#important to note this is on a PER SYSTEM basis
#format here is:
#  [frames per rep]*number of systems 
#                 or
#[frames per rep for system 1]+...[[frames per rep for system n]]
#
# order is important but other than that pretty simple thing in theory I can use
# the mdtraj identification to see if the next frame might belong to a different trajectory
# but this seems like something to explore when a thesis isnt due in a month

replicate_frames=((([80]*20)+([160]*10))*2)


inertia_scores=[]
max_clusters=10
cluster_range = range(2, max_clusters + 1)

for k in cluster_range:
    kluster_output=preform_clust(feature_matrix,n=k)
    np.save(f"/zfshomes/lperez/final_thesis_data/kluster_output/kluster_output_{k}clust",kluster_output)
    labels,inertia=kluster_output[2]
    visualization_df=label_iterator(labels,frame_list=replicate_frames)
    traj_view_replicates_vectorized(visualization_df,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/kmeans_{k}clust_vec.png",clustering=False)
    np.savetxt(f"/zfshomes/lperez/final_thesis_data/kmeans_feature_matrices/feature_matrix{k}_clusters",visualization_df,delimiter=",", fmt="%.6f") #saving arrays for formatting thoroughness


#------------------------------------------
#loading in some of our processed labels and visualizing them for final plots
#------------------------------------------

kmeans_labels_2k=np.load("/zfshomes/lperez/final_thesis_data/kluster_output/redone_kluster_labels_2clust.npy",allow_pickle=True).tolist()
replicate_frames=((([80]*20)+([160]*10))*2)
kmeans_2k_viz = label_iterator(kmeans_labels_2k,frame_list=replicate_frames)
traj_view_replicates_10by10(kmeans_2k_viz,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/2k_clust.png",clustering=False)


