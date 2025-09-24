import numpy as np

def test_rmsd_calculation(modeller): #test both cases
    results = modeller.rmsd_from_centers()
    assert isinstance(results, np.ndarray)
    