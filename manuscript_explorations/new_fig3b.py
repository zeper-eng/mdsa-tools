import numpy as np
from mdsa_tools.Analysis import systems_analysis
import os


#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]

#Extract Principal Components or UMAP
Systems_Analyzer = systems_analysis(all_systems)


X_pca,_,_=Systems_Analyzer.reduce_systems_representations() #you could do method=PCA/UMAP here
print('PCA reduction succesful')

from mdsa_tools.Viz import visualize_reduction
import matplotlib.cm as cm
system_labels=[1]*3200+[2]*3200
visualize_reduction(X_pca,color_mappings=system_labels,savepath=os.getcwd()+'/new_fig3b',cmap=cm.plasma_r)

