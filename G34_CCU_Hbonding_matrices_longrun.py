
import mdtraj as md
#------------------------------------------
#Loading in Trajectories
# Note: This is best if done with concatenated 
# trajectories containing multiplel runs sequentially for farther
# down the pipeline
#------------------------------------------

CCU_GCU_Concatenated='/zfshomes/lperez/fingerprint/cpptraj_testdata/CCU_Concatenated_Trajectories/long_concatenated_GCU.mdcrd'
CCU_GCU_Topology="/home66/kscopino/AMBER22/CODONS/CCUGCU_G34/TLEAP/5JUP_N2_GCU_nowat.prmtop"

CCU_GCU_mdtrajectory = md.load(CCU_GCU_Concatenated,top=CCU_GCU_Topology)

#------------------------------------------
# Creating necessary attributes.
# Note: This was kept seperate as, at one point while helping samviht out
# with his work I did not realize, tools like creating atom-residue dictionaries
# being an easily applicable function would be so useful. Thus, I did away
# with the object oriented approach and in the meantime I have a "Functon Warehouse"
#------------------------------------------
from Data_gen_hbond import create_attributes
GCU_dictionary,GCU_Array=create_attributes(CCU_GCU_mdtrajectory)

#------------------------------------------
# Creating necessary attributes.
#------------------------------------------
from Data_gen_hbond import Process_trajectory
CCU_GCU_Trajectory_array=Process_trajectory(CCU_GCU_mdtrajectory,GCU_Array,GCU_dictionary)

#------------------------------------------------
# Saving original array before further processing
#------------------------------------------------
import numpy as np
np.save("/zfshomes/lperez/final_thesis_data/long_CCU_GCU_Trajectory_array",CCU_GCU_Trajectory_array)

#-----------------------------------------------------------
# Filtering for only unrestrained residues and saving again
#-----------------------------------------------------------

from t_a_Manipulation import filter_res
from Convenience import unrestrained_residues #just a list of int 1 indexed residue indexes

unrestrained_CCU_GCU_Trajectory_array=filter_res(CCU_GCU_Trajectory_array,residues_to_filter=unrestrained_residues)
np.save("/zfshomes/lperez/final_thesis_data/long_unrestrained_average_CCU_GCU_Trajectory_array",unrestrained_CCU_GCU_Trajectory_array)


