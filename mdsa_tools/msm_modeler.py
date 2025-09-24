import numpy as np
import os
import pandas as pd

class MSM_Modeller:
    """
    tiny helper for poking at candidate states and building a very simple msm
    from an embedding space (think: pca/umap coords + k-means labels).
    """

    def __init__(self, labels=None, centers=None, reduced_coordinates=None, frame_scale=None):
        """init with labels/centers/coords/frame sizes—nothing fancy."""
        self.labels = labels
        self.centers = centers
        self.frame_scale = frame_scale
        self.reduced_coordinates = reduced_coordinates

    ###########################################################################
    # candidate state evaluation
    ###########################################################################

    def rmsd_from_centers(self, X=None, labels=None, centers=None):
        X = X if X is not None else self.reduced_coordinates
        labels = labels if labels is not None else self.labels
        centers = centers if centers is not None else self.centers
        results = []
        for k in np.unique(labels):
            m = (labels == k)
            d = np.linalg.norm(X[m] - centers[int(k)], axis=1)
            rmsd = float(np.sqrt(np.mean(d**2)))
            results.append((int(k), rmsd))
        return np.array(results)

    def evaluate_cohesion_slidingwindow(
        self,
        labels=None,
        centers=None,
        reduced_coordinates=None,
        frame_scale=None,
        step_size=None,
    ):
        reduced_coordinates = reduced_coordinates if reduced_coordinates is not None else self.reduced_coordinates
        frame_scale = frame_scale if frame_scale is not None else self.frame_scale
        step_size = step_size if step_size is not None else 10
        labels = labels if labels is not None else self.labels
        centers = centers if centers is not None else self.centers
        # ... rest unchanged ...

    def evaluate_cohesion_shrinkingwindow(
        self,
        labels=None,
        centers=None,
        reduced_coordinates=None,
        frame_scale=None,
        step_size=None,
    ):
        reduced_coordinates = reduced_coordinates if reduced_coordinates is not None else self.reduced_coordinates
        frame_scale = frame_scale if frame_scale is not None else self.frame_scale
        step_size = step_size if step_size is not None else 10
        labels = labels if labels is not None else self.labels
        centers = centers if centers is not None else self.centers
        # ... rest unchanged ...

    ###########################################################################
    # implied timescales + ck test
    ###########################################################################

    def compute_implied_timescales(self, lags, labels=None, frame_list=None, n_timescales=None):
        labels = labels if labels is not None else self.labels
        frame_list = frame_list if frame_list is not None else self.frame_scale
        n_timescales = n_timescales if n_timescales is not None else 10
        # ... rest unchanged ...

    def chapman_kolmogorov_test(self, labels=None, frame_list=None, lag=None, steps=None):
        labels = labels if labels is not None else self.labels
        lag = lag if lag is not None else 30
        frame_list = frame_list if frame_list is not None else self.frame_scale
        steps = steps if steps is not None else 4
        # ... rest unchanged ...

    ###########################################################################
    # transition probability matrix
    ###########################################################################

    def create_transition_probability_matrix(self, labels=None, frame_list=None, lag=None):
        labels = labels if labels is not None else self.labels
        frame_list = frame_list if frame_list is not None else self.frame_scale
        lag = lag if lag is not None else 1
        # ... rest unchanged ...

    def extract_stationary_states(self, final_transition_prob_matrix=None):
        final_transition_prob_matrix = (
            final_transition_prob_matrix
            if final_transition_prob_matrix is not None
            else self.create_transition_probability_matrix()
        )
        # ... rest unchanged ...

    def evaluate_Chapman_Kolmogorov(
        self,
        transition_probability_matrix=None,
        n=None,
        labels=None,
        original_lag=None,
    ):
        transition_probability_matrix = (
            transition_probability_matrix
            if transition_probability_matrix is not None
            else self.create_transition_probability_matrix()
        )
        original_lag = original_lag if original_lag is not None else 1
        n = n if n is not None else 4
        labels = labels if labels is not None else self.labels
        # ... rest unchanged ...
