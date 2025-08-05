from utilities.Convenience import unrestrained_residues
from utilities.Analysis import systems_analysis
import os
import numpy as np

##############################################################################
#Now lets test with bahrats original data as I need to make sure this is good#
##############################################################################

AP_system=np.load('/Users/luis/Desktop/workspace/test_systems/AP_system.npy')
APL_system=np.load('/Users/luis/Desktop/workspace/test_systems/APL_system.npy')
P_system=np.load('/Users/luis/Desktop/workspace/test_systems/P_system.npy')
PL_system=np.load('/Users/luis/Desktop/workspace/test_systems/PL_system.npy')

bahrats_trajectories=[AP_system,APL_system,P_system,PL_system]

bahrats_analyzer=systems_analysis(bahrats_trajectories)

bahrats_analyzer.cluster_system_level(outfile_path='/Users/luis/Desktop/workspace/test_output/systems_kmeans/baharat_')
bahrats_analyzer.reduce_systems_representations(outfile_path='/Users/luis/Desktop/workspace/test_output/PCA/bahrat_')
bahrats_analyzer.cluster_embeddingspace(outfile_path='/Users/luis/Desktop/workspace/test_output/cluster_embeddingspace/bahrat_',max_clusters=4,elbow_or_sillohuette='sillohuette')