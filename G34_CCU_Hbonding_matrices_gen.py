
import mdtraj as md

#------------------------------------------
#Loading in a trajectories                -
#------------------------------------------

ten_frame = '/zfshomes/lperez/summer2025/SBTA_progression/PDBs/CCU_GCU_10frames.mdcrd'
CCU_GCU_Topology = "/home66/kscopino/AMBER22/CODONS/CCUGCU_G34/TLEAP/5JUP_N2_GCU_nowat.prmtop"

CCU_GCU_mdtrajectory = md.load(CCU_GCU_Concatenated,top=CCU_GCU_Topology)

#------------------------------------------
# Creating necessary attributes           -
#------------------------------------------

from Data_gen_hbond import create_attributes
GCU_dictionary,GCU_Array=create_attributes(CCU_GCU_mdtrajectory)
CGU_dictionary,CGU_Array=create_attributes(CCU_CGU_mdtrajectory)

#------------------------------------------------
# Saving original array before further processing
#------------------------------------------------
import numpy as np
np.save("/zfshomes/lperez/final_thesis_data/redone_CCU_GCU_Trajectory_array",CCU_GCU_Trajectory_array)
np.save("/zfshomes/lperez/final_thesis_data/redone_CCU_CGU_Trajectory_array",CCU_CGU_Trajectory_array)

#-----------------------------------------------------------
# Filtering for only unrestrained residues and saving again
#-----------------------------------------------------------

from t_a_Manipulation import filter_res
from Convenience import unrestrained_residues #just a list of int 1 indexed residue indexes


unrestrained_CCU_GCU_Trajectory_array=filter_res(CCU_GCU_Trajectory_array,residues_to_filter=unrestrained_residues)
unrestrained_CCU_CGU_Trajectory_array=filter_res(CCU_CGU_Trajectory_array,residues_to_filter=unrestrained_residues)


np.save("/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_GCU_Trajectory_array",unrestrained_CCU_GCU_Trajectory_array)
np.save("/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_CGU_Trajectory_array",unrestrained_CCU_CGU_Trajectory_array)


