from dim_reduction import peform_PCA_opt
import numpy as np
from t_a_Manipulation import replicates_to_featurematrix

#load in our trajectories
redone_CCU_GCU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)

print(redone_CCU_GCU_fulltraj.shape,redone_CCU_CGU_fulltraj.shape)

redone_arrays=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
redone_feature_matrix = replicates_to_featurematrix(arrays=redone_arrays)

peform_PCA_opt(redone_feature_matrix)