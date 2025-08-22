from mdsa_tools.Analysis import systems_analysis
import numpy as np
import matplotlib.cm as cm
import os
import pandas as pd
from mdsa_tools.msm_modeler import MSM_Modeller as msm


#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)

from mdsa_tools.Viz import visualize_reduction
persys_frame_list=((([80] * 20) + ([160] * 10)))
persys_frame_short=([80] * 20) 
persys_frame_long= ([160] * 10)

#For the paper we move forward with systems representations
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
Systems_Analyzer = systems_analysis(systems_representations=all_systems,replicate_distribution=persys_frame_list)
Systems_Analyzer.replicates_to_featurematrix()
X_pca,_ ,_=Systems_Analyzer.reduce_systems_representations()

candidate_states_per_system=Systems_Analyzer.cluster_embeddingspace()

coordinates=[X_pca[0:3200,:],X_pca[3200:,:]]

for i in range(len(candidate_states_per_system)):
    labels,centers= candidate_states_per_system[i][0],candidate_states_per_system[i][1]
    b = np.arange(3200) 

    only_long  = (1600 <= b) 
    only_short = (1600 > b) 

    only_short_labels=labels[only_short]
    only_long_labels=labels[only_long]

    current_coordinates_short=coordinates[i][only_short,:]
    current_coordinates_long=coordinates[i][only_long,:]
    

    print(only_short_labels.shape)
    print(only_long_labels.shape)

    print(current_coordinates_short.shape)
    print(current_coordinates_long.shape)

    onlyshort_modeler=msm(only_short_labels,centers,current_coordinates_short,persys_frame_short)
    onlylong_modeler=msm(only_long_labels,centers,current_coordinates_long,persys_frame_long)

    onlyshort_results_shrinking=onlyshort_modeler.evaluate_cohesion_shrinkingwindow(step_size=20)
    onlyshort_results_sliding=onlyshort_modeler.evaluate_cohesion_slidingwindow(step_size=20)

    onlylong_results_shrinking=onlylong_modeler.evaluate_cohesion_shrinkingwindow(step_size=20)
    onlylong_results_sliding=onlylong_modeler.evaluate_cohesion_slidingwindow(step_size=20)

    onlyshort_results_shrinking.to_csv(f'system_{i}onlyshort_results_shrinking.csv')
    onlyshort_results_sliding.to_csv(f'system_{i}onlyshort_results_sliding.csv')
    onlylong_results_shrinking.to_csv(f'system_{i}onlylong_shrinkingresults.csv')
    onlylong_results_sliding.to_csv(f'system_{i}onlylong_slidingresults.csv')


os._exit(0)


MSM_modeler=msm(optimal_k_elbow_labels,centers_elbow,X_pca,persys_frame_list)


RMSD_dataframe_sliding=MSM_modeler.evaluate_cohesion_slidingwindow(step_size=10)
RMSD_dataframe_shrinking=MSM_modeler.evaluate_cohesion_shrinkingwindow(step_size=10)

RMSD_dataframe_sliding.to_csv('./RMSD_dataframe_sliding.csv')
RMSD_dataframe_shrinking.to_csv('./RMSD_dataframe_shrinking.csv')


RMSD_dataframe.to_csv('RMSD_dataframe.csv')
