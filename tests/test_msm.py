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
