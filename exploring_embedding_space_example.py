from utilities.Analysis import systems_analysis
import numpy as np
import matplotlib.cm as cm

#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]

#Extract Principal Components or UMAP
Systems_Analyzer = systems_analysis(all_systems)

X_pca,_,_=Systems_Analyzer.reduce_systems_representations(method='PCA',n_components=2) #PCA
umap_embedding=Systems_Analyzer.reduce_systems_representations(method='UMAP', min_dist=.2, n_neighbors=6000) #UMAP


#Lets virst visualize the embedding space
from utilities.Viz import visualize_reduction
substitute_kmeans_labels=(([1]*3200)+([2]*3200))
visualize_reduction(X_pca,color_mappings=substitute_kmeans_labels,savepath='/Users/luis/Desktop/workspacetwo/test_output/PCA/justPCA',cmap=cm.magma_r)


#Visualize replicates in embedding space 
frame_list=((([80] * 20) + ([160] * 10)) * 2)
from utilities.Viz import highlight_reps_in_embeddingspace
highlight_reps_in_embeddingspace(reduced_coordinates=X_pca,outfilepath='test_output/per_rep/G34CCU_PCA_')#PCA
highlight_reps_in_embeddingspace(reduced_coordinates=X_pca,outfilepath='test_output/per_rep/G34CCU_UMAP_')#UMAP

#Contour embedding space 
from utilities.Viz import contour_embedding_space
contour_embedding_space('test_output/contour/contour_test_PCA',X_pca)#PCA
contour_embedding_space('test_output/contour/contour_test_PCA',umap_embedding)#UMAP

#Cluster embedding space (PCA)
results=Systems_Analyzer.cluster_individual_systems_in_embeddingspace()

iterator=1
for i in results:
    visualize_reduction(X_pca,color_mappings=i,savepath='/Users/luis/Desktop/workspacetwo/test_output/PCA/system_{1}_embeddingspacecluster_',cmap=cm.magma_r)
    iterator+=1

#Cluster embedding space (UMAP)
results=Systems_Analyzer.cluster_individual_systems_in_embeddingspace(reduced_data=umap_embedding)

iterator=1
for i in results:
    visualize_reduction(umap_embedding,color_mappings=i,savepath='/Users/luis/Desktop/workspacetwo/test_output/UMAP/system_{1}_embeddingspacecluster_',cmap=cm.magma_r)
    iterator+=1
    



