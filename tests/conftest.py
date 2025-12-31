from pathlib import Path
import pytest
from mdsa_tools.Data_gen_hbond import TrajectoryProcessor
from mdsa_tools.Convenience import unrestrained_residues
from mdsa_tools.Analysis import systems_analysis
import numpy as np

# ----------------------------------------------------------------
# Trajectory Data Cases
# ----------------------------------------------------------------
DATA = Path(__file__).parent / "data" / "trajectories"
CASES = [
    (DATA / "CCU_GCU_10frames.mdcrd", DATA / "5JUP_N2_GCU_nowat.prmtop"),
    (DATA / "CCU_CGU_10frames.mdcrd", DATA / "5JUP_N2_CGU_nowat.prmtop"),
]

@pytest.fixture(scope="session", params=CASES, ids=["GCU", "CGU"])
def processor(request):
    traj, top = request.param
    return TrajectoryProcessor(traj, top)

@pytest.fixture(scope="session")
def systems(processor):
    return processor.create_system_representations()

@pytest.fixture(scope="session")
def filtered(processor):
    return processor.create_filtered_representations(residues_to_keep=unrestrained_residues)

# ----------------------------------------------------------------
# Ribosome Analysis Systems
# ----------------------------------------------------------------
@pytest.fixture(scope="session")
def analysis_systems():
    trajs = [
        ("CCU_GCU_10frames.mdcrd", "5JUP_N2_GCU_nowat.prmtop"),
        ("CCU_CGU_10frames.mdcrd", "5JUP_N2_CGU_nowat.prmtop"),
    ]
    arrays = []
    for traj, top in trajs:
        tp = TrajectoryProcessor(DATA / traj, DATA / top)
        current_array = tp.create_filtered_representations(residues_to_keep=unrestrained_residues)
        arrays.append(current_array)
    return arrays

@pytest.fixture(scope="session")
def analyzer(analysis_systems):
    sa = systems_analysis(analysis_systems)  # ribosome set
    sa.replicates_to_feature_matrix()
    return sa

# ----------------------------------------------------------------
# MDshare brings in extra Systems (separate from ribosome)
# ----------------------------------------------------------------
'''
Citation for using data provided by MDAnalysis
Oliver Beckstein, Richard Gowers, Irfan Alibay, Shujie Fan, Lily Wang, & Micaela Matta. 
(2023). MDAnalysis/MDAnalysisData: 0.9.0 (release-0.9.0). Zenodo. https://doi.org/10.5281/zenodo.10058664

Seyler, Sean; Beckstein, Oliver (2017): Molecular dynamics trajectory for benchmarking MDAnalysis. figshare. Fileset. doi: 10.6084/m9.figshare.5108170.v1
'''
RES = Path(__file__).parent / "data" / "trajectories"

ADK_TOP = RES / "adk_topology.pdb"
ADK_TRAJ = RES / "adk_10frames.dcd"

import mdtraj as md
@pytest.fixture(scope="session")
def external_systems():
    full_traj = md.load(str(ADK_TRAJ), top=str(ADK_TOP))
    tp = TrajectoryProcessor(preloaded_trajectory=full_traj, one_indexed=False)
    return tp.create_system_representations()

@pytest.fixture(scope="session")
def external_analyzer(external_systems):
    external_analyzer = systems_analysis([external_systems])  
    external_analyzer.replicates_to_feature_matrix()
    return external_analyzer

# ----------------------------------------------------------------
# Embeddings, Colors, Labels
# ----------------------------------------------------------------
@pytest.fixture(scope="session")
def small_embedding():
    column_one = np.arange(0, 6400, 1)
    column_two = column_one.copy()
    test_coordinates = np.column_stack((column_one, column_two))
    assert test_coordinates.shape == (6400, 2)
    return test_coordinates

@pytest.fixture(scope="session")
def discrete_colors():
    return np.array([0, 0, 1, 1], dtype=int)

@pytest.fixture(scope="session")
def legend_labels_map():
    return {0: "#1f77b4", 1: "#ff7f0e"}

@pytest.fixture(scope="session")
def simple_labels_and_frames():
    labels = np.arange(0, 6400, 1)
    frame_list = ((([80] * 20) + ([160] * 10)) * 2)
    return labels, frame_list

# ----------------------------------------------------------------
# Cpptraj Import Fixtures
# ----------------------------------------------------------------
CPPTRAJ_CASES = [
    (
        Path(__file__).parent / "data" / "cpptraj_fake_data" / "Break_On_Fake_Cpptraj_Data.dat",
        Path(__file__).parent / "data" / "trajectories" / "5JUP_N2_GCU_nowat.prmtop",
    )
]

from mdsa_tools.Cpptraj_import import cpptraj_hbond_import

@pytest.fixture(scope="session", params=CPPTRAJ_CASES, ids=["GCU"])
def importer(request):
    datfile, top = request.param
    residues = [95,230,232,234,235,236,237,238,239,240,241,242,243,244,245,246]
    return cpptraj_hbond_import(datfile, top, residues)

@pytest.fixture(scope="session", params=CPPTRAJ_CASES, ids=["GCU"])
def alternatefeaturematrix(importer):
    feature_matrix = importer.iterate_frames()
    return feature_matrix

@pytest.fixture(scope="session")
def precomputed_analyzer(alternatefeaturematrix, importer):
    """
    Build an analyzer directly from the precomputed feature matrix produced
    by the cpptraj importer, using the same residue index ordering.
    """
    sa = systems_analysis(
        precomputed_feature_matrix=alternatefeaturematrix,
        res_indexes_for_precomputedmatrix=importer.res_of_interest
    )
    return sa


# ----------------------------------------------------------------
# Visualization Fixtures
# ----------------------------------------------------------------
import matplotlib.pyplot as plt

@pytest.fixture(scope="session")
def less_than_256_bin_colormappings():
    return np.concatenate((np.full(3200, 1), np.full(3200, 2)))

@pytest.fixture(scope="session")
def emptyplotting_space():
    fig, ax = plt.subplots()
    return (fig, ax)

@pytest.fixture(scope="session")
def kvals_and_silscores():
    k_vals = np.array([2, 3, 4, 5, 6])
    scores = np.array([0.12, 0.28, 0.41, 0.57, 0.49])
    return k_vals, scores

@pytest.fixture(scope="session")
def kvals_and_inertiascores():
    k_vals = np.array([2, 3, 4, 5, 6])
    inertia_scores = np.array([500, 320, 220, 180, 170])
    return k_vals, inertia_scores

@pytest.fixture(scope="session")
def rankedweights_df(analyzer):
    return analyzer.create_PCA_ranked_weights()

@pytest.fixture(scope="session")
def rmsd_df():
    import pandas as pd
    return pd.DataFrame({
        "window": [1, 2, 1, 2],
        "rmsd": [0.5, 0.6, 0.7, 0.8],
        "cluster": ["A", "A", "B", "B"]
    })

# ----------------------------------------------------------------
# MSM Fixtures
# ----------------------------------------------------------------
from mdsa_tools.subdomain_explorations import subdomain_explorations

DATA_MSM = Path(__file__).parent / "data" / "klust"
CASES_MSM = [
    (
        DATA_MSM / "GCU_coordinates_kluster_labels_5clust.npy",
        DATA_MSM / "GCU_sil_centers.npy",
        DATA_MSM / "GCU_coordinates.npy",
    ),
    (
        DATA_MSM / "GCU_coordinates_kluster_labels_2clust.npy",
        DATA_MSM / "CGU_sil_centers.npy",
        DATA_MSM / "CGU_coordinates.npy",
    ),
]
IDS = ["GCU", "CGU"]

@pytest.fixture(scope="session", params=CASES_MSM, ids=IDS)
def generic_labels_and_centers(request):
    labels_path, centers_path, coords_path = request.param
    labels = np.load(labels_path).astype(int)
    centers = np.load(centers_path)
    coords = np.load(coords_path)
    return labels, centers, coords

@pytest.fixture(scope="session", params=CASES_MSM, ids=IDS)
def modeller(request):
    labels_path, centers_path, coords_path = request.param
    labels = np.load(labels_path).astype(int)
    centers = np.load(centers_path)
    reduced_coordinates = np.load(coords_path)
    frame_scale = [len(reduced_coordinates)]

    uniq = np.unique(labels)
    if not np.array_equal(uniq, np.arange(len(uniq))):
        remap = {old: i for i, old in enumerate(uniq)}
        labels = np.vectorize(remap.get)(labels)

    return subdomain_explorations(labels, centers, reduced_coordinates, frame_scale)
