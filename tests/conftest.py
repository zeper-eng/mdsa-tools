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
# that all of our dat works wether its filtered or the original full matrices
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
# === Fixtures for visualization tests ===

@pytest.fixture()
def small_embedding():
    # 4 points in 2D square
    return np.array([[0.0, 0.0],
                     [1.0, 0.0],
                     [0.0, 1.0],
                     [1.0, 1.0]], dtype=float)

@pytest.fixture()
def discrete_colors():
    # Two categories: 0 and 1
    return np.array([0, 0, 1, 1], dtype=int)

@pytest.fixture()
def legend_labels_map():
    # Map discrete label -> color (as expected by visualize_reduction)
    return {0: "#1f77b4", 1: "#ff7f0e"}  # blue / orange


@pytest.fixture()
def simple_labels_and_frames():
    # labels length must equal sum(frame_list)
    labels = np.array([0, 1, 0, 2, 2], dtype=int)  # 5 frames total
    frame_list = [3, 2]  # replicate 0: 3 frames, replicate 1: 2 frames
    return labels, frame_list
