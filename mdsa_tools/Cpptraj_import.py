import numpy as np
import mdtraj as md
from typing import Tuple, Dict

class cpptraj_hbond_import():

    def __init__(self,filepath,topology):
        ''' Init takes just the filepath to the desires data and then the topology

        Parameters
        ----------



        Returns
        -------



        Examples
        --------



        Notes
        -----



        '''
        self.indices=self.extract_headers(filepath)
        self.data=np.loadtxt(filepath, skiprows=1, usecols=range(1, len(self.indices)+1), dtype=int)
        self.topology = md.load_topology(topology) 

        return
     
    def extract_headers(self,filepath):
        '''Smaller module for importing files from cpptraj 

        Parameters
        ----------



        Returns
        -------



        Notes
        -----



        Examples
        --------


        '''
        filepath = filepath if filepath is not None else None

        lines=[]
        indices=[]
        with open(filepath,'r') as infile:
            for line in infile:
                lines.append(line.split())

        for col_header in lines[0]:
            if col_header !='#Frame':
                res1 = col_header.split('_')[1].split('@')[0]
                res2 = col_header.split('_')[2].split('@')[0]
                indices.append((int(res1),int(res2)))
        return indices

    def create_cpptraj_attributes(self,data,topology,granularity=None):
            '''returns atom to residue dictionary and template array for processing

            Parameters
            ----------
            trajectory:mdtraj.Trajectory

            Returns
            -------
            atom_to_residue:Dict, atom_to_residue[atom_index]=residue_index
                Dictionary containing atom to residue mappings

            template_array: np.ndarray, shape=(n_frames,n_residues,n_residues)
                returns array containing adjacency matrices for every frame. Shape is dependent on residues in trajectory and number of frames.

            Examples
            --------
            

            Notes
            -----
            This atom to residue dictionary is important as the function we will use for extracting hydrogen bonding information
            returns hydrogen bonds at the atomic level, and we need it at the residue level for this particular "systems" 
            representation. 

            The template array is so we only create one datastructure to modify later improving efficiency.

            '''

            granularity = granularity if granularity is not None else 'residue'

            #Make atom to residue dictionary 

            #Create adjacency matrix, set first row and column as residue indices, and multiply to match the number of frames
            
            topology = md.load_topology(topology) if topology is not None else self.topology

            if granularity == 'residue':

                indexes=[residue.resSeq+1 for residue in topology.residues]
                empty_array = np.zeros(shape=(len(indexes)+1,len(indexes)+1)) 

                empty_array[0,1:]=indexes
                empty_array[1:,0]=indexes

                template_array=np.repeat(empty_array[np.newaxis,:, :], data.shape[0], axis=0)

                return template_array
            
    def create_systems_rep(self,data=None,topology=None,indices=None):
        '''Filling in the matrix

        Parameters
        ----------

        Returns
        -------

        Notes
        -----

        Examples
        --------

        '''
        topology = topology if topology is not None else self.topology
        data = data if data is not None else self.data
        indices = indices if indices is not None else self.indices

        template_array=self.create_cpptraj_attributes(data,topology)

        iterator=0

        for col in data.T: #simply transpose so we are going column wise instead
            current_pair=indices[iterator]

            if current_pair[0]!=current_pair[1]:
                template_array[:,current_pair[0]-1,current_pair[1]-1]=col

            iterator+=1
        
        return template_array
    

if __name__ == '__main__':

        
    from mdsa_tools.Data_gen_hbond import TrajectoryProcessor as tp
    import numpy as np
    import os
    from mdsa_tools.Convenience import unrestrained_residues

    topology = '../PDBs/5JUP_N2_GCU_nowat.prmtop'
    traj = '../PDBs/CCU_GCU_10frames.mdcrd' 

    test_trajectory = tp(trajectory_path=traj,topology_path=topology)


    print("succesfully loaded current PDB from test")

    os._exit(0)

    test_atomic_system=test_trajectory.create_system_representations(test_trajectory.trajectory,granularity='atom')
    print(test_atomic_system.shape)

    test_atomic_system_no_indexes=test_atomic_system[0,1:,1:]
    print(test_atomic_system_no_indexes[test_atomic_system_no_indexes!=0])

    print('test running just the datagen file')


    #########################################
    #In house test with our own trajectories#
    #########################################

    #load in and test trajectory
    system_one_topology = '../PDBs/5JUP_N2_CGU_nowat.prmtop'
    system_one_trajectory = './CCUGCU_G34_full.mdcrd' 
    system_two_topology = '../PDBs/5JUP_N2_GCU_nowat.prmtop'
    system_two_trajectory = './CCUCGU_G34_full.mdcrd' 

    print('run one')
    test_trajectory_one = tp(trajectory_path=system_one_trajectory,topology_path=system_one_topology)
    print("tp made")
    test_system_one_ = test_trajectory_one.create_filtered_representations(residues_to_keep=unrestrained_residues)
    print("systems made")
    np.save('./full_sampling_GCU',test_system_one_)
    print(" made")


    del test_trajectory_one, test_system_one_ 

    test_trajectory_two = tp(trajectory_path=system_two_trajectory,topology_path=system_two_topology)
    print("tp made")
    test_system_two_ = test_trajectory_two.create_filtered_representations(residues_to_keep=unrestrained_residues)
    print("systems made")
    np.save('./full_sampling_CGU',test_system_two_)
    print(" made")


    #now that its loaded in try to make object
    print("intializing creation made")
    print("finished creation")

    del test_trajectory_two, test_system_two_ 








    