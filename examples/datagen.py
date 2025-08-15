from mdsa_tools.Data_gen_hbond import trajectory as traj
import numpy as np
import os

#########################################
#In house test with our own trajectories#
#########################################

#load in and test trajectory
system_one_topology = '/Users/luis/Desktop/workspace/PDBs/5JUP_N2_CGU_nowat.prmtop'
system_one_trajectory = '/Users/luis/Desktop/workspace/PDBs/CCU_CGU_10frames.mdcrd' 

system_two_topology = '/Users/luis/Desktop/workspace/PDBs/5JUP_N2_GCU_nowat.prmtop'
system_two_trajectory = '/Users/luis/Desktop/workspace/PDBs/CCU_GCU_10frames.mdcrd' 

test_trajectory_one = traj(trajectory_path=system_one_trajectory,topology_path=system_one_topology)
test_trajectory_two = traj(trajectory_path=system_two_trajectory,topology_path=system_two_topology)


#now that its loaded in try to make object
test_system_one_ = test_trajectory_one.create_system_representations()
test_system_two_ = test_trajectory_two.create_system_representations()

print(test_system_one_[0])
print(test_system_two_[0])

os._exit(0)


np.save('/Users/luis/Desktop/workspacetwo/example_systems/test_system_one',test_system_one_)
np.save('/Users/luis/Desktop/workspacetwo/example_systems/test_system_two',test_system_two_)






