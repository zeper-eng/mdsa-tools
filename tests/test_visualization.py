# Use non-GUI backend during tests (safe on CI)
import matplotlib
matplotlib.use("Agg", force=True)
from pathlib import Path
import mdsa_tools.Viz as vz


def test_visualize_reduction_largebins_saves_file(tmp_path, small_embedding):
    out = tmp_path / "viz_continuous.png"
    vz.visualize_reduction(
        embedding_coordinates=small_embedding,
        color_mappings=None,      # triggers continuous colormap branch
        savepath=out,        # function expects a file path here
        title="Test Continuous",
        cmap=None,
        axis_one_label=None,
        axis_two_label=None,
        cbar_label=None,
        gridvisible=False,
    )

    assert out.exists(), "visualize_reduction did not create the output file"
    assert out.stat().st_size > 0, "output image is empty"


def test_visualize_reduction_saves_file(tmp_path, small_embedding, less_than_256_bin_colormappings):
    out = tmp_path / "viz_discrete.png"
    vz.visualize_reduction(
        embedding_coordinates=small_embedding,
        color_mappings=less_than_256_bin_colormappings,  # triggers discrete colormap branch
        savepath=out,
        title="Test Discrete",
        cmap=None,
        axis_one_label=None,
        axis_two_label=None,
        cbar_label=None,
        gridvisible=False,
    )

    assert out.exists(), "visualize_reduction did not create the output file"
    assert out.stat().st_size > 0, "output image is empty"


def test_replicatemap_from_labels_saves_png(tmp_path, simple_labels_and_frames):
    labels, frame_list = simple_labels_and_frames
    out = tmp_path 

    vz.replicatemap_from_labels(
        labels=labels,
        frame_list=frame_list,
        savepath=out,   
        title="Replicate Map Test",
        xlabel="Frame",
        ylabel="Replicate",
        cmap=None,
    )

    assert out.exists(), "replicatemap_from_labels did not create the output file"
    assert out.stat().st_size > 0, "output image is empty"



def test_continuous_colorbar_branch(tmp_path, small_embedding):
    out = tmp_path / "viz_auto_continuous.png"
    vz.visualize_reduction(
        embedding_coordinates=small_embedding,
        color_mappings=None,
        savepath=out,
        title=None,
        cmap=None,
        axis_one_label=None,
        axis_two_label=None,
        cbar_label=None,
        gridvisible=False,
    )

    assert out.exists()

def test_discrete_colorbar_branch(tmp_path, small_embedding,less_than_256_bin_colormappings):
    out = tmp_path / "viz_auto_discrete.png"
    vz.visualize_reduction(
        embedding_coordinates=small_embedding,
        color_mappings=less_than_256_bin_colormappings,
        savepath=out,
        title=None,
        cmap=None,
        axis_one_label=None,
        axis_two_label=None,
        cbar_label=None,
        gridvisible=False,
    )
    
    assert out.exists()

def test_plot_silhouette_scores(tmp_path,kvals_and_silscores):
    kvals,scores=kvals_and_silscores
    out_prefix = tmp_path    
    
    best_k = vz.plot_sillohette_scores(
        cluster_range=kvals,
        silhouette_scores=scores,
        outfile_path=str(out_prefix),
        title="Sil scores",
        xlabel="k",
        ylabel="score",
    )

    assert best_k==5

def test_plot_elbow_scores(tmp_path,kvals_and_inertiascores):
    k_vals, inertia_scores=kvals_and_inertiascores
    out_prefix = tmp_path / "elbow_test"  
    
    best_k = vz.plot_elbow_scores(
        cluster_range=k_vals,
        inertia_scores=inertia_scores,
        outfile_path=str(out_prefix),
        title="Elbow Test",
        xlabel="k",
        ylabel="Inertia"
    )

    # Verify the returned optimal k
    assert best_k == 5 



    

