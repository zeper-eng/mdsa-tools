import numpy as np
import pandas as pd 
from utilities.t_a_Manipulation import filter_res
from time import time 
from utilities.Viz import format_for_visualization, visualize


start=time()
#Filler script because my jupyter installation decided to break
unrestrained_CCU_GCU_Trajectory=np.load('/zfshomes/lperez/final_thesis_data/unrestrained_CCU_GCU_Trajectory_array.npy')
unrestrained_CCU_CGU_Trajectory=np.load('/zfshomes/lperez/final_thesis_data/unrestrained_CCU_CGU_Trajectory_array.npy')


#filtering for our residues of interest

residues_to_filter = [411,422] #seperate component

unrestrained_CCU_GCU_Trajectory_6675 = filter_res(unrestrained_CCU_GCU_Trajectory,residues_to_filter)
unrestrained_CCU_CGU_Trajectory_6675 = filter_res(unrestrained_CCU_CGU_Trajectory,residues_to_filter)


#Creating our dataframe for easy analysis and outputting to our terminal (annoyingly)

hbonddf_6677 = pd.DataFrame({
    'frames': range(unrestrained_CCU_GCU_Trajectory_6675.shape[0]),  # Creates a sequence from 0 to 3199
    'GCU_hbonds_6675': unrestrained_CCU_GCU_Trajectory_6675[:, 1, 2],
    'CGU_hbonds_6675': unrestrained_CCU_CGU_Trajectory_6675[:, 1, 2]
})

print(f"total execution time:{time()-start}\n\n")

print(hbonddf_6677.sort_values(by='GCU_hbonds_6675',ascending=False))
#print(hbonddf_6677.sort_values(by='GCU_hbonds_6675',ascending=False).head(10))
#lets see quickly how they vary in terms of the actual simulation

GCU_6677_overtraj=unrestrained_CCU_GCU_Trajectory_6675[:,1,2]
CGU_6677_overtraj=unrestrained_CCU_CGU_Trajectory_6675[:,1,2]

print(np.max(GCU_6677_overtraj))
print(np.max(CGU_6677_overtraj))

segment_configurations = [
        {'start': 0, 'end': 80, 'repeat': 20, 'padding': 80},      # First 20 replicates
        {'start': 1600, 'end': 1760, 'repeat': 10,'padding': 0},                # First 10 replicates
        {'start': 3200, 'end': 3280, 'repeat': 20, 'padding': 80}, # Next 20 replicates
        {'start': 4800, 'end': 4960, 'repeat': 10,'padding': 0}                 # Last 10 replicates
    ]
sixsixsevenseven = np.concatenate((GCU_6677_overtraj,CGU_6677_overtraj))

kmeans_grand_finale=format_for_visualization(array=sixsixsevenseven,segment_configs=segment_configurations)
visualize(array=kmeans_grand_finale,name='/zfshomes/lperez/thesis_figures/pairwise/fourelevemforutwentytwo.png')


########################################################################################################################################################################################################################
import os
os._exit(0)
#I34 case
I34_unrestrained_CCU_GCU_Trajectory=np.load('/zfshomes/lperez/final_thesis_data/I34_unrestrained_CCU_GCU_Trajectory_array.npy')
I34_unrestrained_CCU_CGU_Trajectory=np.load('/zfshomes/lperez/final_thesis_data/I34_unrestrained_CCU_CGU_Trajectory_array.npy')


I34_residues_to_filter = [66,75]
I34_unrestrained_CCU_GCU_Trajectory_6675 = filter_res(I34_unrestrained_CCU_GCU_Trajectory,residues_to_filter)
I34_unrestrained_CCU_CGU_Trajectory_6675 = filter_res(I34_unrestrained_CCU_CGU_Trajectory,residues_to_filter)

I34_hbonddf_6677 = pd.DataFrame({
    'frames': range(unrestrained_CCU_GCU_Trajectory_6675.shape[0]),  # Creates a sequence from 0 to 3199
    'GCU_hbonds_6675': unrestrained_CCU_GCU_Trajectory_6675[:, 1, 2],
    'CGU_hbonds_6675': unrestrained_CCU_CGU_Trajectory_6675[:, 1, 2]
})

#print(hbonddf_6677)
print(I34_hbonddf_6677.sort_values(by='GCU_hbonds_6675',ascending=False).head(10))
