# tests/test_analysis.py
import numpy as np
import os

# ------------------------------------------------------------
# Precomputed analyzer: Using our ribosome example
# ------------------------------------------------------------

def test_feature_matrix_shape(analyzer):
    # sanity: feature matrix exists and has expected number of rows (samples/frames)
    fm = analyzer.feature_matrix
    assert fm.shape[0] == 18

def test_proper_PCA_reduction_output(analyzer):
    # PCA returns coords (same n_samples), component loadings, and explained variance ratio
    X_pca, weights, var_ratio = analyzer.reduce_systems_representations(method="PCA")
    assert X_pca.shape[0] == analyzer.feature_matrix.shape[0]   # same number of rows as input
    assert isinstance(var_ratio, np.ndarray) and var_ratio.shape[0] == 2  # 2D reduction
    assert weights.shape[1] == analyzer.feature_matrix.shape[1]  # loadings span all features

def test_proper_UMAP_reduction_output(analyzer):
    # UMAP returns a 2D embedding with one row per sample
    umap_coordinates = analyzer.reduce_systems_representations(method="UMAP", n_neighbors=5)
    assert umap_coordinates.shape[0] == analyzer.feature_matrix.shape[0]
    assert umap_coordinates.shape[1] == 2

def test_system_clustering(analyzer):
    # k-means wrapper (with silhouette+elbow) returns label vectors and centers for each method
    optimal_k_silhouette_labels, optimal_k_elbow_labels, centers_silhouette, centers_elbow = analyzer.perform_kmeans(max_clusters=5)

    # labels must align with samples
    assert optimal_k_silhouette_labels.shape[0] == analyzer.feature_matrix.shape[0], "silhouette clustering labels dont match n_samples"
    assert optimal_k_elbow_labels.shape[0] == analyzer.feature_matrix.shape[0], "elbow clustering labels dont match n_samples"

    # centers must have same feature dimension as input
    assert centers_silhouette.shape[1] == analyzer.feature_matrix.shape[1], "silhouette cluster centers wrong dimension"
    assert centers_elbow.shape[1] == analyzer.feature_matrix.shape[1], "elbow cluster centers wrong dimension"

    # both strategies should pick ≥2 clusters for this dataset
    assert centers_silhouette.shape[0] >= 2, "silhouette clustering found too few clusters"
    assert centers_elbow.shape[0] >= 2, "elbow clustering found too few clusters"
    
def test_pca_ranked_weights(analyzer):
    # ensures ranked-weights table is well-formed and magnitudes are non-negative
    analyzer.reduce_systems_representations(method='UMAP')  # run one reduction to populate state (if needed)
    ranked_weights = analyzer.create_PCA_ranked_weights()
    assert ranked_weights.shape[0] == analyzer.feature_matrix.shape[1], "incorrect number of comparisons in ranked_weights creation"

    # required columns present
    for col in ["Comparisons","PC1_Weights","PC2_Weights","PC1_magnitude","PC2_magnitude"]:
        assert col in ranked_weights.columns
    
    features = analyzer.feature_matrix.shape[1]
    # magnitudes are absolute values
    assert (ranked_weights["PC1_magnitude"].values >= 0).all()
    assert (ranked_weights["PC2_magnitude"].values >= 0).all()
    # one weight per feature for each PC
    assert ranked_weights["PC1_Weights"].shape[0] == features
    assert ranked_weights["PC2_Weights"].shape[0] == features

    # comparisons formatted like "i-j" (used later by MDCircos)
    assert ranked_weights["Comparisons"].str.contains(r"^\d+-\d+$").all()

    return

def test_replicates_to_feature_matrix_accepts_single_array(analysis_systems):
    # the method should also work when passed a single array (not a list)
    from mdsa_tools.Analysis import systems_analysis as sas
    single_array = analysis_systems[0]
    analyzer = sas([single_array])
    Feature_Matrix = analyzer.replicates_to_feature_matrix()
    assert np.count_nonzero(Feature_Matrix) > 0  # not all zeros

def test_perform_clust_opt_fixed_k_returns_shapes(tmp_path, analyzer):
    # fixed-k clustering path returns labels (n_samples) and centers (k × n_features)
    Feature_Matrix = analyzer.replicates_to_feature_matrix()
    labels, centers = analyzer.perform_clust_opt(outfile_path=str(tmp_path) + os.sep, data=Feature_Matrix, k=2)
    assert labels.shape[0] == Feature_Matrix.shape[0]
    assert centers.shape == (2, Feature_Matrix.shape[1])

def test_perform_kmeans_k_path(tmp_path, analyzer):
    # explicit-k kmeans on precomputed feature matrix
    Fm = analyzer.feature_matrix
    labels, centers = analyzer.perform_kmeans(outfile_path=str(tmp_path) + os.sep, data=Fm, k=2)
    assert labels.shape[0] == Fm.shape[0]
    assert centers.shape[0] == 2

# ------------------------------------------------------------
# Precomputed analyzer: Using our third MDAnalysis system
# ------------------------------------------------------------

def test_external_feature_matrix_shape(external_analyzer):
    # sanity: feature matrix exists and has expected number of rows (samples/frames)
    fm = external_analyzer.feature_matrix
    assert fm.shape[0] == 10

def test_external_proper_PCA_reduction_output(external_analyzer):
    # PCA returns coords (same n_samples), component loadings, and explained variance ratio
    X_pca, weights, var_ratio = external_analyzer.reduce_systems_representations(method="PCA")
    assert X_pca.shape[0] == external_analyzer.feature_matrix.shape[0]   # same number of rows as input
    assert isinstance(var_ratio, np.ndarray) and var_ratio.shape[0] == 2  # 2D reduction
    assert weights.shape[1] == external_analyzer.feature_matrix.shape[1]  # loadings span all features

def test_external_proper_UMAP_reduction_output(external_analyzer):
    # UMAP returns a 2D embedding with one row per sample
    umap_coordinates = external_analyzer.reduce_systems_representations(method="UMAP", n_neighbors=5)
    assert umap_coordinates.shape[0] == external_analyzer.feature_matrix.shape[0]
    assert umap_coordinates.shape[1] == 2

def test_external_system_clustering(external_analyzer):
    # k-means wrapper (with silhouette+elbow) returns label vectors and centers for each method
    optimal_k_silhouette_labels, optimal_k_elbow_labels, centers_silhouette, centers_elbow = external_analyzer.perform_kmeans(max_clusters=5)

    # labels must align with samples
    assert optimal_k_silhouette_labels.shape[0] == external_analyzer.feature_matrix.shape[0], "silhouette clustering labels dont match n_samples"
    assert optimal_k_elbow_labels.shape[0] == external_analyzer.feature_matrix.shape[0], "elbow clustering labels dont match n_samples"

    # centers must have same feature dimension as input
    assert centers_silhouette.shape[1] == external_analyzer.feature_matrix.shape[1], "silhouette cluster centers wrong dimension"
    assert centers_elbow.shape[1] == external_analyzer.feature_matrix.shape[1], "elbow cluster centers wrong dimension"

    # both strategies should pick ≥2 clusters for this dataset
    assert centers_silhouette.shape[0] >= 2, "silhouette clustering found too few clusters"
    assert centers_elbow.shape[0] >= 2, "elbow clustering found too few clusters"
    
def test_external_pca_ranked_weights(external_analyzer):
    # ensures ranked-weights table is well-formed and magnitudes are non-negative
    external_analyzer.reduce_systems_representations(method='UMAP')  # run one reduction to populate state (if needed)
    ranked_weights = external_analyzer.create_PCA_ranked_weights()
    assert ranked_weights.shape[0] == external_analyzer.feature_matrix.shape[1], "incorrect number of comparisons in ranked_weights creation"

    # required columns present
    for col in ["Comparisons","PC1_Weights","PC2_Weights","PC1_magnitude","PC2_magnitude"]:
        assert col in ranked_weights.columns
    
    features = external_analyzer.feature_matrix.shape[1]
    # magnitudes are absolute values
    assert (ranked_weights["PC1_magnitude"].values >= 0).all()
    assert (ranked_weights["PC2_magnitude"].values >= 0).all()
    # one weight per feature for each PC
    assert ranked_weights["PC1_Weights"].shape[0] == features
    assert ranked_weights["PC2_Weights"].shape[0] == features

    # comparisons formatted like "i-j" (used later by MDCircos)
    assert ranked_weights["Comparisons"].str.contains(r"^\d+-\d+$").all()

    return


def test_external_perform_clust_opt_fixed_k_returns_shapes(tmp_path, external_analyzer):
    # fixed-k clustering path returns labels (n_samples) and centers (k × n_features)
    Feature_Matrix = external_analyzer.replicates_to_feature_matrix()
    labels, centers = external_analyzer.perform_clust_opt(outfile_path=str(tmp_path) + os.sep, data=Feature_Matrix, k=2)
    assert labels.shape[0] == Feature_Matrix.shape[0]
    assert centers.shape == (2, Feature_Matrix.shape[1])

def test_external_perform_kmeans_k_path(tmp_path, external_analyzer):
    # explicit-k kmeans on precomputed feature matrix
    Fm = external_analyzer.feature_matrix
    labels, centers = external_analyzer.perform_kmeans(outfile_path=str(tmp_path) + os.sep, data=Fm, k=2)
    assert labels.shape[0] == Fm.shape[0]
    assert centers.shape[0] == 2

# ------------------------------------------------------------
# Precomputed analyzer: uses alternatefeaturematrix from importer
# ------------------------------------------------------------

def test_precomputed_feature_matrix_shape(precomputed_analyzer, importer):
    """Same as original shape test but on the precomputed path."""
    fm = precomputed_analyzer.feature_matrix
    assert fm.ndim == 2 and fm.shape[0] > 0 and fm.shape[1] > 0

    # Columns must be nC2 for the exact residues used by the importer
    n = len(importer.res_of_interest)
    E = n * (n - 1) // 2
    assert fm.shape[1] == E


def test_precomputed_proper_PCA_reduction_output(precomputed_analyzer):
    X_pca, weights, var_ratio = precomputed_analyzer.reduce_systems_representations(method="PCA")
    assert X_pca.shape[0] == precomputed_analyzer.feature_matrix.shape[0]
    assert isinstance(var_ratio, np.ndarray) and var_ratio.shape[0] == 2
    assert weights.shape[1] == precomputed_analyzer.feature_matrix.shape[1]


def test_precomputed_proper_UMAP_reduction_output(precomputed_analyzer):
    emb = precomputed_analyzer.reduce_systems_representations(method="UMAP", n_neighbors=5)
    assert emb.shape[0] == precomputed_analyzer.feature_matrix.shape[0]
    assert emb.shape[1] == 2


def test_precomputed_system_clustering_fixed_k(precomputed_analyzer, tmp_path):
    X = precomputed_analyzer.feature_matrix
    labels, centers = precomputed_analyzer.perform_kmeans(outfile_path=str(tmp_path) + "/", data=X, k=2)
    assert labels.shape[0] == X.shape[0]
    assert centers.shape == (2, X.shape[1])


def test_precomputed_pca_ranked_weights(precomputed_analyzer, importer):
    precomputed_analyzer.reduce_systems_representations(method="PCA")
    df = precomputed_analyzer.create_PCA_ranked_weights()
    # columns present
    for col in ["Comparisons", "PC1_Weights", "PC2_Weights", "PC1_magnitude", "PC2_magnitude"]:
        assert col in df.columns
    # sizes
    features = precomputed_analyzer.feature_matrix.shape[1]
    assert df.shape[0] == features
    # magnitudes non-negative
    assert (df["PC1_magnitude"].values >= 0).all()
    assert (df["PC2_magnitude"].values >= 0).all()
    # comparison label format
    assert df["Comparisons"].str.contains(r"^\d+-\d+$").all()

# ------------------------------------------------------------
# Miscellaneous new tests for various quantifications
# ------------------------------------------------------------

def test_perform_optimizedUMAP_local(analyzer):
    fillerdf=analyzer.perform_optimized_UMAP_local(max_neighbors=5,min_neighbors=2,eval_n=2)
    expected_cols = {"n_neighbors", "min_dist", "trustworthiness score", "bubble_size"}

    #check basic dataframe attributes are true
    assert expected_cols.issubset(fillerdf.columns)
    assert fillerdf.shape[1] == 4
    assert np.issubdtype(fillerdf["n_neighbors"].dtype, np.integer)
    assert fillerdf["min_dist"].between(0.0, 1.0).all()
    assert fillerdf["trustworthiness score"].between(0, 1.0).all()

    return

def test_perform_optimizedUMAP_global(analyzer):
    fillerdf=analyzer.perform_optimized_UMAP_global(max_neighbors=5,min_neighbors=2)
    expected_cols = {"n_neighbors", "min_dist", "pearson_r", "bubble_size"}

    #check basic dataframe attributes are true
    assert expected_cols.issubset(fillerdf.columns)
    assert fillerdf.shape[1] == 4
    assert np.issubdtype(fillerdf["n_neighbors"].dtype, np.integer)
    assert fillerdf["min_dist"].between(0.0, 1.0).all()
    assert fillerdf["pearson_r"].between(-1.0, 1.0).all()

    return