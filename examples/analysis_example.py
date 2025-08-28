from mdsa_tools.Convenience import unrestrained_residues
from mdsa_tools.Analysis import systems_analysis
import os
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


#We seperate out the data generation phase and assume you generated and saved your data prior
redone_CCU_GCU_fulltraj=np.load('/Users/marcarseneiradukunda/Documents/Wesleyan_academics/Summer_2025/Testing_Luis_code/workspace/data_gen_output/test_system_two.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/marcarseneiradukunda/Documents/Wesleyan_academics/Summer_2025/Testing_Luis_code/workspace/data_gen_output/test_system_two.npy',allow_pickle=True)
all_systems=[redone_CCU_GCU_fulltraj]

Systems_Analyzer = systems_analysis(all_systems)

#Clustering and visualizing clusters 
optimal_k_silhouette_labels,optimal_k_elbow_labels,centers_sillohuette,centers_elbow = Systems_Analyzer.cluster_system_level(outfile_path='/Users/marcarseneiradukunda/Documents/Wesleyan_academics/Summer_2025/Testing_Luis_code/workspace/test_output/systems_kmeans/',max_clusters=25)
print('clustering succesfully completed')


Systems_Analyzer.reduce_systems_representations(outfile_path='/Users/marcarseneiradukunda/Documents/Wesleyan_academics/Summer_2025/Testing_Luis_code/workspace/test_output/PCA/test_',colormappings=optimal_k_silhouette_labels) #you could do method=PCA/UMAP here
print('PCA reduction succesful')
Systems_Analyzer.cluster_embeddingspace(outfile_path='/Users/marcarseneiradukunda/Documents/Wesleyan_academics/Summer_2025/Testing_Luis_code/workspace/test_output/cluster_embeddingspace/',max_clusters=10,elbow_or_sillohuette='sillohuette')
print('Embedding space clustering succesfully completed')



