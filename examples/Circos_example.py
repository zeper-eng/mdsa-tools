from mdsa_tools.Analysis import systems_analysis
import os
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

#Load in previously generated data
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]

#the goal of this is to analyze multiple differnt structures so naturally we need two different systems
Systems_Analyzer = systems_analysis(all_systems)


############################################
#Trying out our new MDCircos visualizations#
############################################
PCA_ranked_weights=Systems_Analyzer.create_PCA_ranked_weights()
from mdsa_tools.Viz import create_MDcircos_from_weightsdf
create_MDcircos_from_weightsdf(PCA_ranked_weights)
os._exit(0)

'''

The above is a general use case but, since we have a clear division of our two systems across the midpoint
of the first principal component we decided to take the negative weightings and visualize them by themselves
as well as visualize the positive weightings by themselves. This is why the Circos visualization includes
min max scaling even tho the magnitudes are technically supposed to add up to 1 i.e. already scaled!

'''

from mdsa_tools.Viz import mdcircos_graph,make_MDCircos_object

PC1_magnitudes=PCA_ranked_weights[['Comparisons','PC1_magnitude','PC1_Weights']]

PC1_positive=PC1_magnitudes[PC1_magnitudes['PC1_Weights']>0]
PC1_negative=PC1_magnitudes[PC1_magnitudes['PC1_Weights']<0]

PC1_positive_mag=PC1_positive[['Comparisons','PC1_magnitude']]
PC1_negative_mag=PC1_negative[['Comparisons','PC1_magnitude']]


create_MDcircos_from_weightsdf(PC1_positive_mag,'/Users/luis/Desktop/workspacetwo/test_output/circos/PC1_non_G_leading')
create_MDcircos_from_weightsdf(PC1_negative_mag,'/Users/luis/Desktop/workspacetwo/test_output/circos/PC1_G_leading')

print(PC1_positive_mag)
print(PC1_negative_mag)

os._exit(0)