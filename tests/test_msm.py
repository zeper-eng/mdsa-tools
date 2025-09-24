import numpy as np
import pandas as pd

def test_rmsd_calculation(modeller): #test both cases
    results = modeller.rmsd_from_centers()
    assert isinstance(results, np.ndarray)
    assert len(np.unique(results[:, 0])) == len(np.unique(modeller.labels))


def test_slidingwindow_basic(modeller):
    df = modeller.evaluate_cohesion_slidingwindow(step_size=10)

    # It should be a DataFrame with required columns
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"cluster", "rmsd", "window"}

    # Number of unique windows should match floor(max_frames / step_size)
    n_expected = (max(modeller.frame_scale) // 10)
    assert df["window"].nunique() == n_expected

    # Each cluster should appear in every window
    for win in df["window"].unique():
        subset = df[df["window"] == win]
        clusters_in_window = set(subset["cluster"].unique())
        assert clusters_in_window.issubset(np.unique(modeller.labels))
        assert len(clusters_in_window) > 0  # shouldn’t be empty

def test_shrinkingwindow_basic(modeller):
    df = modeller.evaluate_cohesion_shrinkingwindow(step_size=10)

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"cluster", "rmsd", "window"}

    n_expected = (max(modeller.frame_scale) // 10)
    assert df["window"].nunique() == n_expected

    # Windows should monotonically drop frames → total rows shrink each time
    counts = df.groupby("window").size().values
    assert all(counts[i] >= counts[i+1] for i in range(len(counts)-1))

def test_sliding_vs_shrinking_overlap(modeller):
    df_slide = modeller.evaluate_cohesion_slidingwindow(step_size=10)
    df_shrink = modeller.evaluate_cohesion_shrinkingwindow(step_size=10)

    # Both should report same set of clusters
    assert set(df_slide["cluster"].unique()) == set(df_shrink["cluster"].unique())

    # Both should have at least 1 window
    assert df_slide["window"].nunique() > 0
    assert df_shrink["window"].nunique() > 0

def test_step_size_effect(modeller):
    df_small = modeller.evaluate_cohesion_slidingwindow(step_size=5)
    df_large = modeller.evaluate_cohesion_slidingwindow(step_size=20)

    # Smaller step → more windows
    assert df_small["window"].nunique() > df_large["window"].nunique()

############
# Moving into some MSM tests
############
def _P_from_headered(T):
    """Helper: strip the 1-row/1-col header."""
    return T[1:, 1:]


def test_transition_matrix_shape_and_stochastic(modeller):
    T = modeller.create_transition_probability_matrix(lag=1)
    P = _P_from_headered(T)

    # square, nonnegative
    n_states = len(np.unique(modeller.labels))
    assert P.shape == (n_states, n_states)
    assert np.all(P >= 0)

    # each row with outgoing transitions sums to ~1
    row_sums = P.sum(axis=1)
    has_outgoing = row_sums > 0
    assert np.allclose(row_sums[has_outgoing], 1.0, atol=1e-8)


def test_transition_matrix_zero_when_lag_geq_replicate_len(modeller):
    big_lag = max(modeller.frame_scale)  # >= length of each replicate
    T = modeller.create_transition_probability_matrix(lag=big_lag)
    P = _P_from_headered(T)
    assert np.allclose(P, 0.0)


def test_transition_matrix_defaults_equal_explicit(modeller):
    T1 = modeller.create_transition_probability_matrix(lag=2)
    T2 = modeller.create_transition_probability_matrix(
        labels=modeller.labels,
        frame_list=modeller.frame_scale,
        lag=2,
    )
    assert np.allclose(T1, T2)


def test_stationary_distribution_is_left_eigenvector(modeller):
    T = modeller.create_transition_probability_matrix(lag=1)
    P = _P_from_headered(T)
    pi = modeller.extract_stationary_states(T)

    # Valid probability vector
    assert np.isfinite(pi).all()
    assert np.all(pi >= -1e-12)  # tiny numerical wiggle
    assert np.isclose(pi.sum(), 1.0, atol=1e-10)

    # Left eigenvector property: pi * P = pi
    assert np.allclose(pi @ P, pi, atol=1e-6)


def test_chapman_kolmogorov_shapes_and_bounds(modeller):
    res = modeller.chapman_kolmogorov_test(lag=1, steps=3)
    # keys are 1..3 and each value is (T_pred, T_direct)
    assert set(res.keys()) == {1, 2, 3}

    n_states = len(np.unique(modeller.labels))
    for k, (T_pred, T_direct) in res.items():
        assert T_pred.shape == (n_states, n_states)
        assert T_direct.shape == (n_states, n_states)
        # probabilities should lie in [0,1] up to numerical jitter
        assert (T_pred >= -1e-12).all() and (T_pred <= 1 + 1e-12).all()
        assert (T_direct >= -1e-12).all() and (T_direct <= 1 + 1e-12).all()


def test_evaluate_CK_scalar_nonnegative(modeller):
    # Use a COPY so we don't mutate the modeller's stored matrix in-place
    T = modeller.create_transition_probability_matrix(lag=1).copy()
    delta = modeller.evaluate_Chapman_Kolmogorov(
        transition_probability_matrix=T,
        original_lag=1,
        n=3,
    )
    assert isinstance(delta, float)
    assert np.isfinite(delta) and (delta >= 0.0)


def test_implied_timescales_basic(modeller):
    lags = [1, 2, 3]
    out = modeller.compute_implied_timescales(lags=lags, n_timescales=5)

    # correct keys and array shapes
    assert set(out.keys()) == set(lags)
    n_states = len(np.unique(modeller.labels))
    max_ts = max(len(v) for v in out.values())
    assert max_ts <= max(1, n_states - 1)

    # finite, positive timescales
    for lag, arr in out.items():
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1
        assert np.isfinite(arr).all()
        # With abs(eigs) used, timescales should be positive
        assert (arr > 0).all()


def test_implied_timescales_respects_n_timescales(modeller):
    # request only the single slowest
    out = modeller.compute_implied_timescales(lags=[2], n_timescales=1)
    assert 2 in out and isinstance(out[2], np.ndarray)
    assert out[2].shape == (min(1, max(1, len(np.unique(modeller.labels)) - 1)),)


