from utilities.Data_gen_hbond import trajectory as traj
import numpy as np


#########################################
#In house test with our own trajectories#
#########################################

#load in and test trajectory
topology = '/Users/luis/Desktop/workspace/PDBs/5JUP_N2_GCU_nowat.prmtop'
trajectory = '/Users/luis/Desktop/workspace/PDBs/CCU_GCU_10frames.mdcrd' 
test_trajectory = traj(trajectory_path=trajectory,topology_path=topology)

#now that its loaded in try to make object
test_system=test_trajectory.create_system_representations()

print(test_system.shape)
print(test_system[0])






