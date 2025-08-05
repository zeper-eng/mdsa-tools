from utilities.Analysis import systems_analysis
import os
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)

all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
replicate_frames = (([80] * 20) + ([160] * 10)) * 2
substitute_kmeans_labels=(([1]*3200)+([2]*3200))


#the goal of this is to analyze multiple differnt structures so naturally we need two different systems
Systems_Analyzer = systems_analysis(all_systems)


############################################
#Trying out our new MDCircos visualizations#
############################################
PCA_ranked_weights=Systems_Analyzer.create_PCA_ranked_weights()

print(PCA_ranked_weights)

def extract_properties_from_table(pca_table):
    comps = pca_table['Comparisons'].astype(str)

    # split stack and clean
    sides = comps.str.split('-', n=1, expand=True)
    residues = (sides.stack()
                      .str.strip()
                      .dropna()
                      .unique())

    # arc ids are strings
    residues = [str(x) for x in residues]

    PC1_weight_dict = pca_table.set_index('Comparisons')['PC1_magnitude'].to_dict()
    PC2_weight_dict = pca_table.set_index('Comparisons')['PC2_magnitude'].to_dict()
    return residues, PC1_weight_dict, PC2_weight_dict
  


res_indexes,PC1_magnitude_dict,PC2_magnitude_dict = extract_properties_from_table(PCA_ranked_weights)


from utilities.Viz import mdcircos_graph,make_MDCircos_object

pc1_circos_object=make_MDCircos_object(res_indexes)
pc2_circos_object=make_MDCircos_object(res_indexes)

# This is a general use case
mdcircos_graph(pc1_circos_object,PC1_magnitude_dict,'/Users/luis/Desktop/workspacetwo/test_output/circos/PC1_magnitudeviz')
mdcircos_graph(pc2_circos_object,PC2_magnitude_dict,'/Users/luis/Desktop/workspacetwo/test_output/circos/PC2_magnitudeviz')

'''

The above is a general use case but, since we have a clear division of our two systems across the midpoint
of the first principal component we decided to take the negative weightings and visualize them by themselves
as well as visualize the positive weightings by themselves. This is why the Circos visualization includes
min max scaling even tho the magnitudes are technically supposed to add up to 1 i.e. already scaled!

'''

PC1_magnitudes=PCA_ranked_weights[['Comparisons','PC1_magnitude','PC1_Weights']]

PC1_positive=PC1_magnitudes[PC1_magnitudes['PC1_Weights']>0]
PC1_negative=PC1_magnitudes[PC1_magnitudes['PC1_Weights']<0]

PC1_positive_mag=PC1_positive[['Comparisons','PC1_magnitude']]
PC1_negative_mag=PC1_negative[['Comparisons','PC1_magnitude']]

PC1_positive_mag=PC1_positive_mag.set_index('Comparisons')['PC1_magnitude'].to_dict()
PC1_negative_mag=PC1_negative_mag.set_index('Comparisons')['PC1_magnitude'].to_dict()

mdcircos_graph(pc1_circos_object,PC1_positive_mag,'/Users/luis/Desktop/workspacetwo/test_output/circos/PC1_non_G_leading')
mdcircos_graph(pc2_circos_object,PC1_negative_mag,'/Users/luis/Desktop/workspacetwo/test_output/circos/PC1_G_leading')

print(PC1_positive_mag)
print(PC1_negative_mag)

os._exit(0)