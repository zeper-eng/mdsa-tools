import numpy as np
import importlib.resources as ir
from pathlib import Path

from mdsa_tools.Data_gen_hbond import Generate_Networks

def test_data_creation():
    from mdsa_tools.Convenience import unrestrained_residues
    GCU_topology = Path(__file__).parent / "data/trajectories/5JUP_N2_GCU_nowat.prmtop"  
    CGU_topology = Path(__file__).parent / "data/trajectories/5JUP_N2_CGU_nowat.prmtop"  
    CCU_GCU_10frames = Path(__file__).parent / "data/trajectories/CCU_GCU_10frames.mdcrd"  
    CCU_CGU_10frames = Path(__file__).parent / "data/trajectories/CCU_CGU_10frames.mdcrd"  

    CCU_GCU_systems = Generate_Networks(CCU_GCU_10frames,GCU_topology)
    CCU_CGU_systems = Generate_Networks(CCU_CGU_10frames,CGU_topology)
   
    
    # setup empty system representation
    # could be done later but I find the explicit definition is more readable
   
    CCU_GCU_systems.create_filtered_representations(residues_to_filter=unrestrained_residues)
    CCU_CGU_systems.create_filtered_representations(residues_to_filter=unrestrained_residues)

    print(f"system one shape: {CCU_GCU_systems.shape}\nsystem two shape:{CCU_CGU_systems.shape}")
    return
     
if __name__ == "__main__":

    test_data_creation()