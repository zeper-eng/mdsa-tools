from mdsa_tools.Analysis import systems_analysis
import numpy as np
import matplotlib.cm as cm
import os 

#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]

#Preform kmeans clustering
Systems_Analyzer = systems_analysis(all_systems)
optimal_k_silhouette_labels,optimal_k_elbow_labels,centers_sillohuette,centers_elbow=Systems_Analyzer.cluster_system_level('/Users/luis/Desktop/workspacetwo/test_output/systems_kmeans/',max_clusters=4)

#Visualize as a replicate map
from mdsa_tools.Viz import replicatemap_from_labels
frame_list=((([80] * 20) + ([160] * 10)) * 2)
replicatemap_from_labels(optimal_k_silhouette_labels,frame_list)

