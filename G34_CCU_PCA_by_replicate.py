#------------------------------------------
#Functions and paths necessary for this
#------------------------------------------
import sys
sys.path.append("/zfshomes/lperez/final_thesis_scripts/pypure/utilities/")
import numpy as np
from Convenience import test_list_2
from t_a_Manipulation import replicates_to_featurematrix
from Viz import label_iterator,traj_view_replicates_10by10
from dim_reduction import run_PCA

legend_labels=legend_labels = {
    'GCU Short': 'purple',
    'GCU Long (0-80)': 'orange',
    'GCU Long (80-160)': 'green',
    'CGU Short': 'yellow',
    'CGU Long (0-80)': 'blue',
    'CGU Long (80-160)': 'red'
}

#------------------------------------------
#Loading in our files of interest and creating feature matrix
#------------------------------------------

#load in our trajectories
redone_CCU_GCU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
print(redone_CCU_GCU_fulltraj.shape,redone_CCU_CGU_fulltraj.shape)
redone_arrays=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
redone_feature_matrix = replicates_to_featurematrix(arrays=redone_arrays)
X_pca,weights,explained_variance_ratio_=run_PCA(redone_feature_matrix,n=2)


#------------------------------------------------
#In regards to our first two principle components
#------------------------------------------------

from Viz import create_PCA_per_rep

create_PCA_per_rep(X_pca)



import os
os._exit(0)

data_fitted_tolargest_principle_component=X_pca[:,0].astype(list)
print(len(data_fitted_tolargest_principle_component))

#------------------------------------------
#Visualizing with replicates as our distinctions
#------------------------------------------
replicate_frames = (([80] * 20) + ([160] * 10)) * 2

divided_X_PCA = label_iterator(data_fitted_tolargest_principle_component,replicate_frames)

traj_view_replicates_10by10(array=divided_X_PCA,title="Colored by replicate"
                            ,savepath="/zfshomes/lperez/final_thesis_scripts/data_files_created_from_scripts/test_multirep",clustering = False)

