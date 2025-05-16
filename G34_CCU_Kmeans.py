import numpy as np
import sys
sys.path.append("/zfshomes/lperez/final_thesis_scripts/pypure/utilities")
from t_a_Manipulation import replicates_to_featurematrix
from Clustering import preform_clust,preform_clust_opt
from Viz import label_iterator,traj_view_replicates_10by10
import os
from matplotlib import cm


#------------------------------------------------------------
#Loading in our files of interest and creating feature matrix
#------------------------------------------------------------
replicate_frames = (([80] * 20) + ([160] * 10)) * 2

#Load in arrays and make them a list then concatenate them
redone_CCU_GCU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
print(redone_CCU_GCU_fulltraj.shape,redone_CCU_CGU_fulltraj.shape)
redone_arrays=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
redone_feature_matrix = replicates_to_featurematrix(arrays=redone_arrays)


#--------------------------------------------------------------------------------------------
#Calculating k-10 with optimization (shown first up next you can manually create k=2-n plots)
# Note: this is redone with our re-concatenated trajectories
#--------------------------------------------------------------------------------------------
max_clusters=10

redone_optimal_k_silhouette_labels,redone_optimal_k_elbow_labels=preform_clust_opt(redone_feature_matrix,"/zfshomes/lperez/final_thesis_data/kluster_output/redone_",10) #note add prefix for name

redone_silhouette_viz_df=label_iterator(redone_optimal_k_silhouette_labels,frame_list=replicate_frames)
redone_optimal_k_elbow_labels = label_iterator(redone_optimal_k_elbow_labels,frame_list=replicate_frames)
print(redone_silhouette_viz_df.shape,redone_optimal_k_elbow_labels.shape)
traj_view_replicates_10by10(redone_silhouette_viz_df,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/reconcatenated_optimal_silhouette_clust.png",clustering=False,colormap=cm.cool_r)
traj_view_replicates_10by10(redone_optimal_k_elbow_labels,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/reconcatenated_optimal_optimal_elbow_clust.png",clustering=False,colormap=cm.cool_r)


#------------------------------------------
#loading in some of our processed labels and visualizing them for final plots
#------------------------------------------

kmeans_labels_2k=np.load("/zfshomes/lperez/final_thesis_data/kluster_output/redone_kluster_labels_2clust.npy",allow_pickle=True).tolist()
kmeans_2k_viz = label_iterator(kmeans_labels_2k,frame_list=replicate_frames)
traj_view_replicates_10by10(kmeans_2k_viz,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/redone_2k_clust.png",clustering=False,colormap=cm.cool_r)

kmeans_labels_10k=np.load("/zfshomes/lperez/final_thesis_data/kluster_output/redone_kluster_labels_10clust.npy",allow_pickle=True).tolist()
kmeans_2k_viz = label_iterator(kmeans_labels_10k,frame_list=replicate_frames)
traj_view_replicates_10by10(kmeans_labels_10k,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/redone_10k_clust.png",clustering=False,colormap=cm.cool_r)


#------------------------------------------
#Loading in original concatenations
#------------------------------------------

print(replicate_frames)
#Load in arrays and make them a list then concatenate them
CCU_GCU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
CCU_CGU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
print(CCU_GCU_fulltraj.shape,CCU_CGU_fulltraj.shape)
arrays=[CCU_GCU_fulltraj,CCU_CGU_fulltraj]
feature_matrix = replicates_to_featurematrix(arrays=arrays)

#--------------------------------------------------------------------------------------------
#Calculating k-10 with optimization (shown first up next you can manually create k=2-n plots)
#--------------------------------------------------------------------------------------------
max_clusters=10
optimal_k_silhouette_labels,optimal_k_elbow_labels=preform_clust_opt(feature_matrix,"/zfshomes/lperez/final_thesis_data/kluster_output/Original_",10)

silhouette_viz_df=label_iterator(optimal_k_silhouette_labels,frame_list=replicate_frames)
optimal_k_elbow_labels = label_iterator(optimal_k_elbow_labels,frame_list=replicate_frames)
print(silhouette_viz_df.shape,optimal_k_elbow_labels.shape)
print(silhouette_viz_df,optimal_k_elbow_labels)

traj_view_replicates_10by10(silhouette_viz_df,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/original_optimal_silhouette_clust.png",clustering=False,colormap=cm.Greys)
traj_view_replicates_10by10(optimal_k_elbow_labels,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/original_optimal_optimal_elbow_clust.png",clustering=False,colormap=cm.Greys)



os._exit(0)

##---------
# Below is the original method adapted for the sillohuette score and Elbow plot
os._exit(0)
#nested exit here to demonstrate where we are manually ejecting it
##---------

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




