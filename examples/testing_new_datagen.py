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


##############################################################################
#Now lets test with bahrats original data as I need to make sure this is good#
##############################################################################
GCU_mdcrd='/Users/luis/Desktop/workspace/PDBs/CCU_GCU_10frames.mdcrd'
CGU_mdcrd='/Users/luis/Desktop/workspace/PDBs/CCU_CGU_10frames.mdcrd'

GCU_prmtop='/Users/luis/Desktop/workspace/PDBs/5JUP_N2_GCU_nowat.prmtop'
CGU_prmtop='/Users/luis/Desktop/workspace/PDBs/5JUP_N2_CGU_nowat.prmtop'

GCU_traj=traj(trajectory_path=GCU_mdcrd,topology_path=GCU_prmtop)
CGU_traj=traj(trajectory_path=CGU_mdcrd,topology_path=CGU_prmtop)

GCU_system=GCU_traj.create_system_representations()
CGU_system=CGU_traj.create_system_representations()

np.save('/Users/luis/Desktop/workspace/test_systems/AP_system',GCU_system)
np.save('/Users/luis/Desktop/workspace/test_systems/APL_system',CGU_system)



