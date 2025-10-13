import numpy as np
import pytest
from mdsa_tools.Cpptraj_import import cpptraj_hbond_import

def test_general_usecasewithexamples(importer):
    edge_vectors = importer.iterate_frames()
    # expected shape = (n_frames, E) where E = nC2 with n = sliced residues
    n_frames = importer.data.shape[0]
    n = importer.topology.n_residues
    E = n * (n - 1) // 2
    assert edge_vectors.shape == (n_frames, E)

def test_extract_headers_parses_pairs(tmp_path):
    # Header + one data row (cpptraj-style)
    text = "#Frame HB_1@N_2@O HB_2@N_3@O HB_2@N_2@O\n0  1 0 1\n"
    f = tmp_path / "hbonds.dat"
    f.write_text(text)

    # similar to the other one Empty something
    obj = object.__new__(cpptraj_hbond_import)
    out = obj.extract_headers(str(f))
    assert out == [(1, 2), (2, 3), (2, 2)]

def test_edge_listcreation(importer):
    #test that lookup table is indeed about the same size 
    edge_list=importer.edgelist_single_frame()
    lookuptable=importer.lookup_table_from_edgelist()
    assert edge_list.shape[0] == int((lookuptable[1:,1:].shape[0]*(lookuptable[1:,1:].shape[0]-1))/2)

def test_Lookuptable_creation(importer):
    #test that the indexes we are using in the lookup table are the same as the ones we bring in
    lookuptable=importer.lookup_table_from_edgelist()
    assert lookuptable[0,1:].tolist() == importer.res_of_interest
    assert lookuptable[1:,0].tolist() == importer.res_of_interest

def test_Lookuptable_creation(importer):
    #test that the indexes we are using in the lookup table are the same as the ones we bring in
    lookuptable=importer.lookup_table_from_edgelist()
    assert lookuptable[0,1:].tolist() == importer.res_of_interest
    assert lookuptable[1:,0].tolist() == importer.res_of_interest




