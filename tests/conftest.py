from pathlib import Path
import pytest
from mdsa_tools.Data_gen_hbond import TrajectoryProcessor
from mdsa_tools.Convenience import unrestrained_residues
from mdsa_tools.Analysis import systems_analysis
import numpy as np

DATA = Path(__file__).parent / "data" / "trajectories"
CASES = [
    (DATA / "CCU_GCU_10frames.mdcrd", DATA / "5JUP_N2_GCU_nowat.prmtop"),
    (DATA / "CCU_CGU_10frames.mdcrd", DATA / "5JUP_N2_CGU_nowat.prmtop"),
]

@pytest.fixture(scope="session", params=CASES, ids=["GCU", "CGU"])# ids for nice reporting
def processor(request):
    traj, top = request.param
    return TrajectoryProcessor(traj, top)

@pytest.fixture(scope="session")
def systems(processor):
    return processor.create_system_representations()

@pytest.fixture(scope="session")
def filtered(processor):
    return processor.create_filtered_representations(residues_to_keep=unrestrained_residues)


# Theese get remade so we can use them both in a list for analyses
# We can use only the filtered because every test in datagen should have run checks to make sure
# that all of our data works whether its filtered or the original full matrices

@pytest.fixture(scope="session")
def analysis_systems():
    trajs = [
        ("CCU_GCU_10frames.mdcrd", "5JUP_N2_GCU_nowat.prmtop"),
        ("CCU_CGU_10frames.mdcrd", "5JUP_N2_CGU_nowat.prmtop"),
    ]
    arrays = []
    for traj, top in trajs:
        tp = TrajectoryProcessor(DATA / traj, DATA / top)
        current_array=tp.create_filtered_representations(residues_to_keep=unrestrained_residues)
        arrays.append(current_array)
    return arrays


@pytest.fixture(scope="session")
def analyzer(analysis_systems):
    sa = systems_analysis(analysis_systems)  # give both at once
    sa.replicates_to_featurematrix()
    return sa


@pytest.fixture(scope="session")
def small_embedding():
    column_one=np.arange(0,6400,1)
    column_two=column_one.copy()
    test_coordinates=np.column_stack((column_one,column_two))
    assert(test_coordinates.shape==(6400,2))
    return test_coordinates


@pytest.fixture(scope="session")
def discrete_colors():
    # Two categories: 0 and 1
    return np.array([0, 0, 1, 1], dtype=int)


@pytest.fixture(scope="session")
def legend_labels_map():
    # Map discrete label -> color (as expected by visualize_reduction)
    return {0: "#1f77b4", 1: "#ff7f0e"}  # blue / orange


@pytest.fixture(scope="session")
def simple_labels_and_frames():
    
    labels = np.arange(0,6400,1)  #size of our systems
    frame_list = ((([80] * 20) + ([160] * 10))*2) #should add up to above
    
    return labels, frame_list

#########################
#Cpptraj Import Fixtures#
#########################

'''CPPTRAJ cases (more to be included)'''
''' for example simple p53 ten frame out
    from leftover data'''

CPPTRAJ_CASES=[
    (Path(__file__).parent / "data" / "cpptraj_fake_data" / "Break_On_Fake_Cpptraj_Data.dat",Path(__file__).parent / "data" / "trajectories" / '5JUP_N2_GCU_nowat.prmtop')
    ]

from mdsa_tools.Cpptraj_import import cpptraj_hbond_import

@pytest.fixture(scope="session", params=CPPTRAJ_CASES, ids=["GCU"])
def importer(request):
    '''Break-On Cpptraj import by default as test files for creating instance of import object'''
    datfile, top = request.param  
    importer_instance=cpptraj_hbond_import(datfile, top)

    return importer_instance


########################
#Visualization Fixtures#
########################

@pytest.fixture(scope='session')
def less_than_256_bin_colormappings():
    colormapping=np.concatenate((np.full(3200,1),np.full(3200,2)))
    return colormapping

import matplotlib.pyplot as plt
@pytest.fixture(scope='session')
def emptyplotting_space():
    fig, ax = plt.subplots()
    return (fig, ax)

@pytest.fixture(scope='session')
def kvals_and_silscores():
    k_vals   = np.array([2, 3, 4, 5, 6])
    scores   = np.array([0.12, 0.28, 0.41, 0.57, 0.49])  # max at k=5
    return k_vals,scores

@pytest.fixture(scope='session')
def kvals_and_inertiascores():
    k_vals = np.array([2, 3, 4, 5, 6])
    inertia_scores = np.array([500, 320, 220, 180, 170]) 
    return k_vals, inertia_scores

@pytest.fixture(scope='session')
def rankedweights_df(analyzer):
    PCA_ranked_weights = analyzer.create_PCA_ranked_weights()
    return PCA_ranked_weights

@pytest.fixture(scope="session")
def rmsd_df():
    import pandas as pd
    return pd.DataFrame({
        "window": [1, 2, 1, 2],
        "rmsd": [0.5, 0.6, 0.7, 0.8],
        "cluster": ["A", "A", "B", "B"]
    })
