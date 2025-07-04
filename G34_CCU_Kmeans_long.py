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
CCU_GCU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_average_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
CCU_GCU_longtraj=np.load('/zfshomes/lperez/final_thesis_data/long_unrestrained_average_CCU_GCU_Trajectory_array.npy',allow_pickle=True)

arrays=[CCU_GCU_fulltraj,CCU_GCU_longtraj]
feature_matrix = replicates_to_featurematrix(arrays=arrays)


#--------------------------------------------------------------------------------------------
#Calculating k-10 with optimization (shown first up next you can manually create k=2-n plots)
#--------------------------------------------------------------------------------------------


replicate_frames=(([80] * 20) + ([160] * 10)) * 2
print(replicate_frames)
max_clusters=10

long_optimal_k_silhouette_labels,long_optimal_k_elbow_labels=preform_clust_opt(feature_matrix,"/zfshomes/lperez/final_thesis_data/kluster_output//long_",10)

long_silhouette_viz_df=label_iterator(long_optimal_k_silhouette_labels,frame_list=replicate_frames)
long_optimal_k_elbow_labels = label_iterator(long_optimal_k_elbow_labels,frame_list=replicate_frames)

traj_view_replicates_10by10(long_silhouette_viz_df,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/long_optimal_silhouette_clust.png",clustering=False)
traj_view_replicates_10by10(long_optimal_k_elbow_labels,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/long_optimal_optimal_elbow_clust.png",clustering=False)


