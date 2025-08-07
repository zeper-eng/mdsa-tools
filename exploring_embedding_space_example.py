from utilities.Analysis import systems_analysis
import os
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.cm as cm

#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]

#Extract Principal Components or UMAP
Systems_Analyzer = systems_analysis(all_systems)

X_pca,_,_=Systems_Analyzer.reduce_systems_representations(method='PCA',n_components=380)#PCA
embedding=Systems_Analyzer.reduce_systems_representations(feature_matrix=X_pca, method='UMAP', min_dist=.2, n_neighbors=6000) #UMAP

print(X_pca.shape)
print(embedding.shape)

#Lets virst visualize the embedding space
from utilities.Viz import visualize_reduction
substitute_kmeans_labels=(([1]*3200)+([2]*3200))
visualize_reduction(embedding,color_mappings=substitute_kmeans_labels,savepath='/Users/luis/Desktop/workspacetwo/test_output/UMAP/UMAP_mindistpoint2_neighbors6000',cmap=cm.cividis)

#Visualize replicates in embedding space (umap)
frame_list=((([80] * 20) + ([160] * 10)) * 2)
from utilities.Viz import highlight_reps_in_embeddingspace
highlight_reps_in_embeddingspace(data=embedding,outfilepath='test_output/per_rep/G34CCU_PCA+UMAP_')

#Contour embedding space (umap)
from utilities.Viz import contour_embedding_space
contour_embedding_space('test_output/contour/contour_test_one',embedding)

#Cluster embedding space (umap)

