import numpy as np
import mdtraj as md
from typing import Tuple, Dict

def create_attributes(trajectory) -> Tuple[np.ndarray, Dict]:
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

    #Make atom to residue dictionary 
    atom_to_residue = {atom.index:atom.residue.resSeq for atom in trajectory.topology.atoms}

    #Create adjacency matrix, set first row and column as residue indices, and multiply to match the number of frames
    indexes=[residue.resSeq+1 for residue in trajectory.topology.residues]
    empty_array = np.zeros(shape=(len(indexes)+1,len(indexes)+1)) 

    empty_array[0,1:]=indexes
    empty_array[1:,0]=indexes

    template_array=np.repeat(empty_array[np.newaxis,:, :], len(trajectory), axis=0)


    return atom_to_residue,template_array

def Process_trajectory(trajectory,array_template,atom_to_residue)->np.ndarray:
        """returns a call to the now modified array template which has been filled in with data
        
        Processes an individual frame of template array and fills in hydrogen bonding values.
        
        Parameters
        ----------
        trajectory:md.trajectory

        array_template:np.ndarray,shape=(n_residues,n_residues,n_frames)
            This is an empty array of shape (n_residues,n_residues,n_frames) where we have
            n_frames worth of adjacency matrices of size n_residues*n_residues
        
        atom_to_residue:Dict, Dict[atom_index]=residue_index
            Dictionary containing atom to residue index mappings      

        frame:int
            Integer for indexing what frame we are iterating over in array        

            
        Returns
        -------
        array_template:np.ndarray,shape=(n_frames,n_residues,n_residues)
            A reference to the original array. It is updating the same array in memory but, in theory
            it is done for throughness.

        Examples
        --------

        Notes
        -----
        
        """
        for frame in range(0,len(trajectory)):
            #splice our current frame and use axis for indexing
            current_frame=array_template[frame]
            
            #Use Baker Hubard to get donor and acceptor atom indexes then map to residue indexes
            Baker_hubbard=md.baker_hubbard(trajectory[frame])
            donor_atoms,acceptor_atoms=Baker_hubbard[:,0],Baker_hubbard[:,2]
            donor_residues,acceptor_residues=np.array([atom_to_residue[atom] for atom in donor_atoms]),np.array([atom_to_residue[atom] for atom in acceptor_atoms])

            #match atoms to residues and increment in array
            for i in range(donor_residues.shape[0]):
                current_donor,current_acceptor=donor_residues[i]+1,acceptor_residues[i]+1
                if current_donor != current_acceptor:
                    current_frame[current_donor, current_acceptor] += 1
                    current_frame[current_acceptor, current_donor] += 1

        return array_template
