import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")#headless(displayless) backend so plots don't break CI

@pytest.fixture(scope="session")
def rng():
    """A random number generator so that we can run tests from a set seed 
    """
    return np.random.default_rng(0)

@pytest.fixture
def tiny_systems():
    """Return two small synthetic 'systems'; randomized so we know 

    returns [np.ndarray,np.ndarray]
    where each array is of the shape (framesxframes)
    """
    import mdtraj as md
    sys1 = md.load('')  
    sys2 = md.load('')
    return [sys1, sys2]