from mdsa_tools.Analysis import systems_analysis
import numpy as np
import matplotlib.cm as cm
import os
import pandas as pd


#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)

from mdsa_tools.Viz import visualize_reduction
persys_frame_list=((([80] * 20) + ([160] * 10)))

#For the paper we move forward with systems representations
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
Systems_Analyzer = systems_analysis(systems_representations=all_systems,replicate_distribution=persys_frame_list)
Systems_Analyzer.replicates_to_featurematrix()
X_pca,_ ,_=Systems_Analyzer.reduce_systems_representations()
candidate_states_per_system=Systems_Analyzer.cluster_embeddingspace()

#visualize candidate states
from mdsa_tools.Viz import visualize_candidate_states
visualize_candidate_states(candidate_states_per_system,X_pca,cmap=cm.inferno_r)
from mdsa_tools.msm_modeler import MSM_Modeller as msm
MSM_modeler=msm(candidate_states_per_system,X_pca,persys_frame_list)
RMSD_dataframe=MSM_modeler.evaluate_cohesion_shrinkingwindow(step_size=10)


os._exit(0)
RMSD_dataframe.to_csv('RMSD_dataframe.csv')
