import mdtraj as md
import numpy as np
import os




elbow=np.load("/zfshomes/lperez/final_thesis_data/kluster_output/redone_kluster_labels_7clust.npy")
np.savetxt("/zfshomes/lperez/final_thesis_data/kluster_output/csv_redone_kluster_labels_7clust",elbow,delimiter=',')

sillhouette=np.load("/zfshomes/lperez/final_thesis_data/kluster_output/redone_kluster_labels_10clust.npy")
np.savetxt("/zfshomes/lperez/final_thesis_data/kluster_output/csv_long_kluster_labels_10clust",sillhouette,delimiter=',')

import os
os._exit(0)

# Load trajectory
long_traj = md.load("/zfshomes/lperez/mdcrd_1in50_60.mdcrd", 
                    top="/home66/kscopino/AMBER22/CODONS/CCUGCU_G34/TLEAP/5JUP_N2_GCU_nowat.prmtop")

for i in range(len(long_traj)):
    print(i)




GCU_long=md.load('/zfshomes/lperez/fingerprint/cpptraj_testdata/CCU_Concatenated_Trajectories/long_concatenated_GCU.mdcrd',top='/home66/kscopino/AMBER22/CODONS/CCUGCU_G34/TLEAP/5JUP_N2_GCU_nowat.prmtop')
for i in range(len(GCU_long)):
    print(f"current_rep = {i}")
