import numpy as np
from t_a_Manipulation import filter_res
from Viz import plot_adjacency_matrix

car_plusone_Asite_indexes=[94,127,240,408,409,410,423,424,425,426,427,428]
##load in our Trajectory Arrays 
CCU_GCU_Trajectory = np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_GCU_Trajectory_array.npy') 
CCU_CGU_Trajectory = np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_CGU_Trajectory_array.npy')

 = filter_res(CCU_GCU_Trajectory,)
 = filter_res(CCU_CGU_Trajectory,)

 = plot_adjacency_matrix(matrix,name="adjacency_matrix",axis_labels=None,diff_matrix=False)
