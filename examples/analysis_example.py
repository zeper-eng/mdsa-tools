from mdsa_tools.Convenience import unrestrained_residues
from mdsa_tools.Analysis import systems_analysis
import matplotlib.cm as cm
import numpy as np

'''
#We seperate out the data generation phase and assume you generated and saved your data prior
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]

Systems_Analyzer = systems_analysis(all_systems)

#Clustering and visualizing clusters 
optimal_k_silhouette_labels,optimal_k_elbow_labels,centers_sillohuette,centers_elbow = Systems_Analyzer.cluster_system_level(outfile_path='/Users/luis/Desktop/workspacetwo/manuscript_explorations/syskmeans/',max_clusters=25)
print('clustering succesfully completed')


Systems_Analyzer.reduce_systems_representations(outfile_path='/Users/luis/Desktop/workspace/test_output/PCA/test_',colormappings=optimal_k_silhouette_labels) #you could do method=PCA/UMAP here
print('PCA reduction succesful')

Systems_Analyzer.cluster_embeddingspace(outfile_path='/Users/luis/Desktop/workspace/test_output/cluster_embeddingspace/',max_clusters=10,elbow_or_sillohuette='sillohuette')
print('Embedding space clustering succesfully completed')
'''

if __name__=='__main__':
    #We seperate out the data generation phase and assume you generated and saved your data prior
    redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
    redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
    all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]

    Systems_Analyzer = systems_analysis(all_systems)
    Systems_Analyzer.replicates_to_featurematrix()
    hbond_counts_411_422 = Systems_Analyzer.extract_hbond_values(residues=[411,422])
    hbond_counts_412_422 = Systems_Analyzer.extract_hbond_values(residues=[412,422])

    from mdsa_tools.Viz import replicatemap_from_labels
    frames=((([80] * 20) + ([160] * 10)) * 2)
    replicatemap_from_labels(labels=hbond_counts_411_422,frame_list=frames,savepath='replicatemap_discrete_411_422',cmap=cm.magma_r)
    replicatemap_from_labels(labels=hbond_counts_412_422,frame_list=frames,savepath='replicatemap_discrete_412_422',cmap=cm.magma_r)




