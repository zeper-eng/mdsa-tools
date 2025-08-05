from utilities.Convenience import unrestrained_residues
from utilities.Analysis import systems_analysis
import os
import numpy as np

 
filtered_CCU_GCU_Trajectory_array = np.load('/zfshomes/lperez/ba_and_ma/final_thesis_data/redone_unrestrained_CCU_GCU_Trajectory_array.npy') 
filtered_CCU_CGU_Trajectory_array = np.load('/zfshomes/lperez/ba_and_ma/final_thesis_data/redone_unrestrained_CCU_CGU_Trajectory_array.npy')

#the goal of this is to analyze multiple differnt structures so naturally we need two different systems
systems=[filtered_CCU_GCU_Trajectory_array,filtered_CCU_CGU_Trajectory_array]
Systems_Analyzer = systems_analysis(systems)

#Clustering ju=t the systems representations of the trajectories we are interested in
optimal_k_silhouette_labels,optimal_k_elbow_labels,centers_sillohuette,centers_elbow = Systems_Analyzer.cluster_system_level(outfile_path='/Users/luis/Desktop/workspace/test_output/systems_kmeans/',max_clusters=3)
print('clustering succesfully completed')
Systems_Analyzer.reduce_systems_representations(outfile_path='/Users/luis/Desktop/workspace/test_output/PCA/test_',colormappings=optimal_k_silhouette_labels)
print('PCA reduction succesful')
Systems_Analyzer.cluster_embeddingspace(outfile_path='/Users/luis/Desktop/workspace/test_output/cluster_embeddingspace/',max_clusters=10,elbow_or_sillohuette='sillohuette')
print('Embedding space clustering succesfully completed')

