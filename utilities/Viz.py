import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import pycircos.pycircos as py 
import seaborn as sns



#Replicate maps
def label_iterator(labels,frame_list) -> np.ndarray:
    '''returns an array consisting of a re-formatted list of labels through which to view a set.

    Parameters
    ----------
    labels: listlike,shape=(data,)
        A list of labels representing all of the labels from either a Kmeans or other analysis that we would like to
        use for our analysis.

    frame_list: listlike,shape=(data,)
        A list of integers representing the number of frames present in each replicate. This should be in the order
        of which the various versions of the system, and replicates where concatenated. 


    Returns
    -------
    reformatted_labels:numpy.ndarray,shape=(n_replicates,n_frames)
        A final array is returned where each row corresponds to one of our replicates in simulations.
        Each column corresponds to that particular frame in the replicate. If there are replicates of varying lengths
        we pad all to the longest trajectorys length with masked nans.

        
    Examples
    -------- 
    

    Notes
    -----
    

    '''

    label_iterator=0
    reformatted_labels=[]
    max_frames = max(frame_list)

    for rep_length in frame_list:
        current_replicate = np.copy(labels[label_iterator:label_iterator + rep_length]).astype(float)
        padded_replicate = np.pad(current_replicate, (0, max_frames - rep_length), constant_values=np.nan)
        reformatted_labels.append(padded_replicate)
        label_iterator+=rep_length


    reformatted_labels = np.vstack(reformatted_labels)
    #print(reformatted_labels.shape)  # Debugging output
    return reformatted_labels

def traj_view_replicates(array, colors_list=['purple', 'orange', 'green', 'yellow', 'blue', 'red', 'pink', 'cyan', 'grey', 'brown'],
                                clustering=True, savepath='traj_view', title='Clusters per frame', xlabel='Frames', ylabel='Replicates',colormap=cm.plasma_r):
    """Returns None

    A function for visualizing the clusters that all frames of each replicate end up in, with 10x10 squares for clarity.

    Parameters
    ----------
    array: np.ndarray shape=(n_replicates, n_frames)
        Our final array where each row corresponds to a replicate
        and each column corresponds to a frame pertaining to that replicate

    colors_list: list-like, default=['purple', 'orange', 'green', 'yellow', 'blue', 'red', 'pink', 'cyan', 'grey', 'brown']
        A list of colors to visualize clusters by default it contains 10 colors
        which is sufficient for clusters 0-9

    clustering: bool, default=True
        Whether to visualize clusters using predefined colors from colors_list.
        If False the function will use a colormap (e.g. viridis) based on normalized values

    savepath: str, default='traj_view'
        The path where the plot will be saved

    title: str, default='Clusters per frame'
        The title of the plot

    xlabel: str, default='Frames'
        The label for the x-axis

    ylabel: str, default='Replicates'
        The label for the y-axis

    Returns
    -------
    visualized_array: np.ndarray
        A masked heatmap of the trajectory replicates returned for alternative use
    """

    rows, cols = array.shape

      # Example font dict
    labels_font_dict = {
        'family': 'monospace',  # Font family (e.g., 'sans-serif', 'serif', 'monospace')
        'size': 20,             # Font size
        'weight': 'bold',       # Font weight ('normal', 'bold', 'light')
        'style': 'italic',      # Font style ('normal', 'italic', 'oblique')
        'color': 'black',       # Text color
    }

    # Use the Viridis colormap if clustering=False
    if not clustering:
        norm = Normalize(vmin=np.nanmin(array), vmax=np.nanmax(array))
        cmap = colormap if colormap is not None else cm.plasma_r

    #mw wants it black
    # Set figure and background to black
    fig=plt.figure(figsize=(16,12)) 
    fig.tight_layout(pad=0) 

    #plt.gca().set_facecolor('black')
    #fig.set_facecolor('black')  # Set also background of plot to black


# realy the chunk of this other than it just being matplot stuff
#----

    cluster_labels = {} #cluster labels for the is clustering case (although annoyingly enough I think the plasma will be the baseline)
    for i in range(rows):
        for j in range(cols):
            if not np.isnan(array[i, j]):

                x_pos = j 
                y_pos = rows-i-1

                if clustering == True:
                    color_index = int(array[i, j]) % len(colors_list)
                    color = colors_list[color_index]
                    cluster_label = f"Cluster {int(array[i, j])}"

                    # Add label to the dictionary (to make sure it's unique)
                    if cluster_label not in cluster_labels:
                        cluster_labels[cluster_label] = scatter

                elif clustering == False:
                    color = cmap(norm(array[i, j]))  
                    cluster_label = 'Plasma Colormap'

                scatter = plt.scatter(x_pos, y_pos,
                                     color=color,
                                     marker="P", 
                                     s=40)  
#-----

    # Add legend for clustering, and colorbar for not clustering
    if clustering == True:
        plt.legend(cluster_labels.values(), cluster_labels.keys(), title="Clusters", loc='upper left', fontsize='small', markerscale=0.8, bbox_to_anchor=(1.02, 1))    
    
    if clustering == False:
        unique_vals = np.unique(array[~np.isnan(array)]).astype(int)
        bounds = np.append(unique_vals, unique_vals[-1] + 1)  # to define edges between bins

        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(),
                        ticks=unique_vals, shrink=0.8, aspect=30, pad=0.02)

        cbar.set_label(" ", rotation=270, labelpad=10, fontsize=12, fontdict=labels_font_dict)
        cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black')
        cbar.ax.set_yticklabels([str(val) for val in unique_vals])
        
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Add labels, title, ticks, grid, then hide ticks and show every 10
    plt.gca().set_xlabel(xlabel, fontdict=labels_font_dict)
    plt.gca().set_ylabel(ylabel, fontdict=labels_font_dict)
    plt.gca().set_title(title, fontdict=labels_font_dict)

    # Set tick positions based on original dimensions, but spaced out by the scaling factor
    plt.gca().set_xticks(np.arange(0, cols))  
    plt.gca().set_yticks(np.arange(0, rows))  
    plt.gca().set_xticklabels([str(i) if i % 80 == 0 else '' for i in range(cols)], fontdict=labels_font_dict)  
    plt.gca().set_yticklabels([str(i) if i % 30 == 0 else '' for i in range(rows-1, -1, -1)], fontdict=labels_font_dict)  

    plt.gca().tick_params(color='black', labelsize=8,pad=0,axis='both', direction='in')#changed from white

    plt.grid(False)
    plt.savefig(savepath, dpi=300)
    plt.close()

    return

#Optimizing kmeans
def plot_sillohette_scores(cluster_range, silhouette_scores, outfile_path="sillohette_method.png"):
    '''quickly plot sillouhette scores afte running kmeans
    Parameters
    ----------

    Returns
    -------

    Notes
    -----


    Examples
    --------

    '''
    # Optimal k is where the silhouette score is highest

    optimal_k_sil = cluster_range[np.argmax(silhouette_scores)]#return index 

    # Plot Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-')
    plt.axvline(optimal_k_sil, color='red', linestyle='--', linewidth=2, label=f'Optimal k = {optimal_k_sil}')
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for optimal K')
    plt.legend()
    plt.grid(True)
    plt.savefig(outfile_path+'sillohuette_plot', dpi=300)

    return optimal_k_sil
 
def plot_elbow_scores(cluster_range, inertia_scores, outfile_path="elbow_method.png"):
    '''quickly plot sillouhette scores afte running kmeans
    Parameters
    ----------

    Returns
    -------

    Notes
    -----


    Examples
    --------

    '''

    # Find the elbow point using the difference in inertia
    diff = np.diff(inertia_scores)  # First derivative
    diff2 = np.diff(diff)  # Second derivative (change in slope)
    optimal_k = cluster_range[np.argmin(diff2)+1]  # +1 because diff2 is one step shorter

    # Plot the Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, inertia_scores, marker='o', linestyle='-')
    plt.axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k = {optimal_k}')

    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('Elbow Method for Optimal k')
    plt.legend()
    plt.grid(True)

    plt.savefig(outfile_path+'elbow_plot', dpi=300)  # Save the figure


    return optimal_k

#Circos plots
def get_Circos_coordinates(residue, gcircle):
    """helper function for creating coordinates for arc sizes in Circos graph

    Parameters
    ----------
    residue:int,default=None
        A residue index from which to create the current arc (in general you will be iterating
        through residue indexes when using this method)

    gcirlce:py.Gcircle,default=py.Gcircle(figsize=(6,6))
        A Pycircos Gcricle object. By default we use one with a figsize of (6,6).

    

    Return a 4-element tuple telling PyCircos chord_plot()
    to start in the middle of the arc with a radial anchor of 550.



    Returns
    -------
    tuple:coordinates:defualt=(residue, mid_position, mid_position, raxis_position)
        A four member tuple consisting of the positioning needed to create an arc.



    Notes
    -----




    Examples
    --------



    
    """
    arc = gcircle._garc_dict[residue]
    # The "size" is the arc length in PyCircos coordinates
    mid_position = arc.size * 0.5  # center of the arc
    # We'll anchor all chords at radial = 550
    # (this can be changed if your arcs are drawn in a different radial band)
    raxis_position = 550
    return (residue, mid_position, mid_position, raxis_position)

def make_MDCircos_object(residue_indexes):
    """Returns a PyCircos Gcircle scaled arcs

    Returns a PyCircos Gcircle object whose arcs are automatically scaled
    based on how many arcs (residues) there are. Also scales line widths,
    so that very few arcs don't produce huge lines and many arcs don't
    produce lines too thin to see.

    Parameters
    ----------
    residue_indexes : list
        List of residue indices you want as arcs.

    Returns
    -------
    circle : py.Gcircle
        A PyCircos object containing arcs scaled by the number of residues.
    """

    if len(residue_indexes) <= 50:
        circle = py.Gcircle(figsize=(6, 6))
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

        # Add each arc
        for index in residue_indexes:
            circle.add_garc(
                py.Garc(
                    arc_id=str(index),
                    facecolor='#FFFFFF',   
                    edgecolor='#000000',   
                    label=str(index),
                    label_visible=True,
                    labelposition=40,
                    labelsize=6,
                    size=10,        
                    interspace=0,          
                    linewidth=.1          
                )
            )

        circle.set_garcs()


    if len(residue_indexes) >50:
        circle = py.Gcircle(figsize=(10, 10))
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

        # Add each arc
        for index in residue_indexes:
            circle.add_garc(
                py.Garc(
                    arc_id=str(index),
                    facecolor='#FFFFFF',   # White arcs
                    edgecolor='#000000',   # Black outline
                    label=str(index),
                    label_visible=True,
                    labelposition=30,
                    labelsize=2,
                    size=100,         # scaled by number of arcs
                    interspace=4,          # small gap
                    linewidth=.1           # scaled line thickness
                )
            )

    circle.set_garcs()
    return circle

def mdcircos_graph(empty_circle, residue_dict, savepath=os.getcwd()+'mdcircos_graph', scale_factor=5,colormap=cm.magma_r):
    ''' creates and saves a mdcircos graph to a desired output directory

    Parameters
    ----------
    Residue_indexes:list, shape=(n_residues)
        A list of residue indexes pertaining to the residues you would like to use as parts of the circle

    Residue_dict:dict,format:dict['residue']=float(value)
        A dictionary where keys are residue indexes (as strings) and values are floats representing the corresponding 
        edge weights in the adjacency matrix (or another method) used for mapping.

    savepath:str(),default=os.getcwd()+'mdcircos_graph'
        Absolute path to the location and name of the file you would like to save the file. Default is mdcircos_graph in the 
        working directory

    Residue_dict:dict,dict['residue']=float(value)
        A dictionary containing mappings from specific residue indexes *as strings* to their respective edge weights in whatever adjacency matrix
        (or other method) is being used as the basis for mapping.
    
    scale_values:bool,default=False
        A boolean argument meant to give the user the option of using a gradient color map in order to visualize stronger interactions

    Returns
    -------
    None
        Strictly a graphing function the methods can be called individually if youd like to tamper with the
        Circos object further


    Notes
    -----
    This is built as basically a wrapper for another python package so it is a little finicky in its implementation. In theory it should work fine
    with the other two functions and really only needs to be specific in the way that its taking the inputs for.
    
    An important note is that the scale is saved as a seperate colorbar and the values are normalized by min max because it takes
    generally as input the weightings which happen to be too small to really visualize well typically.

    
    Examples
    --------

    '''
   
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import numpy as np


    # Normalize the colors based on the values provided 
    vals = list(residue_dict.values())
    vmin, vmax = min(vals), max(vals)
    color_norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormap if colormap is not None else cm.plasma
    hex_color_map = {k: cmap(color_norm(v)) for k, v in residue_dict.items()}

    # Width normalization on the absolute values via min–max (makes plot aesthetically closer to raw
    # values it is still suggested to use outputted tables for any actual raw analysis of values)
    abs_vals = [abs(v) for v in vals if v != 0]
    min_abs, max_abs = min(abs_vals), max(abs_vals)

    # avoid division by zero if all values are the same magnitude
    denom = max_abs - min_abs if max_abs != min_abs else 1.0

    width_norm = {
        k: (abs(v) - min_abs) / denom
        for k, v in residue_dict.items()
    }

    # 3) Plot chords
    for key, value in residue_dict.items():
        if value == 0:
            continue
        
        res1, res2 = key.split('-')
        arc1 = get_Circos_coordinates(res1, empty_circle)
        arc2 = get_Circos_coordinates(res2, empty_circle)
        color = hex_color_map[key]

        lw = width_norm[key] * scale_factor
        empty_circle.chord_plot(arc1, arc2,
                                linewidth=lw,
                                facecolor=color,
                                edgecolor=color)

    empty_circle.figure.savefig(savepath + ".png",
                                dpi=300, bbox_inches="tight")

    # 4) Separate colorbar (using the original signed range)
    fig_cb, ax_cb = plt.subplots(figsize=(1.5, 4))
    sm = cm.ScalarMappable(cmap=cmap, norm=color_norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=ax_cb)
    ticks = np.linspace(vmin, vmax, num=6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    cbar.set_label("Directional Difference")

    fig_cb.savefig(savepath + "_colorbar.png",
                dpi=300, bbox_inches="tight")
    
    plt.close(fig_cb)

def extract_properties_from_weightsdf(pca_table):

    comps = pca_table['Comparisons'].astype(str)

    # split stack and clean
    sides = comps.str.split('-', n=1, expand=True)
    residues = (sides.stack()
                      .str.strip()
                      .dropna()
                      .unique())

    # arc ids are strings
    residues = [str(x) for x in residues]

    PC1_weight_dict = pca_table.set_index('Comparisons')['PC1_magnitude'].to_dict()
    PC2_weight_dict = pca_table.set_index('Comparisons')['PC2_magnitude'].to_dict()
    return residues, PC1_weight_dict, PC2_weight_dict

def create_MDcircos_from_weightsdf(PCA_ranked_weights,outfilepath='/Users/luis/Desktop/workspacetwo/test_output/circos/'):
    '''Processes Weights table to create MDcircos plots visualizing weightings

    Parameters
    ----------
    PCA_ranked_weights:Pandas.DataFrame,default=None
      A table containing the ranked weights we created as a part of the analysis Systems Analysis
      module. It is expected that it wold contain columns with the headers:

    Returns
    -------

    Notes
    -----



    Examples
    --------



    Notes
    -----


    '''
    res_indexes,PC1_magnitude_dict,PC2_magnitude_dict = extract_properties_from_weightsdf(PCA_ranked_weights)
    pc1_circos_object=make_MDCircos_object(res_indexes)
    pc2_circos_object=make_MDCircos_object(res_indexes)
    mdcircos_graph(pc1_circos_object,PC1_magnitude_dict,outfilepath+'PC1_magnitudeviz')
    mdcircos_graph(pc2_circos_object,PC2_magnitude_dict,outfilepath+'PC2_magnitudeviz') 


#Embeddingspace visualizations
def create_2d_color_mappings(labels=([80]*20)+([160]*10), 
                             colors_list=['purple', 'orange', 'green', 'yellow', 'blue', 'red', 'pink', 'cyan', 'grey', 'brown'], 
                             clustering=True):
    ''' 
    Parameters
    ----------
    labels: list, shape (n_samples), default=([80]*20)+([160]*10)
        A list of integers that help describe how you want to label each sample once they have been reduced to 2 dimensions.

    colors_list: list-like, default=['purple', 'orange', 'green', 'yellow', 'blue', 'red', 'pink', 'cyan', 'grey', 'brown']
        A list of colors that we can use to visualize all of our clusters.

    clustering: bool, default=True
        Whether to assign discrete colors for clusters (True) or use a heatmap-based visualization (False).
    '''

    if clustering == True:
        # Create a dictionary mapping each label to a color
        label_dict = {}
        i = 0
        for label in labels:
            if label not in label_dict:
                label_dict[label] = colors_list[i % len(colors_list)]  # Ensure cycling through colors if necessary
                i += 1
        sample_color_mappings = [label_dict[i] for i in labels]
        return sample_color_mappings

def visualize_reduction(X_pca, color_mappings=None, custom=False, 
                  savepath=os.getcwd(), 
                  title="Principal Component Analysis (PCA) of GCU and CGU Systems", 
                  colors_list=['purple', 'orange', 'green', 'yellow', 'blue', 'red', 'pink', 'cyan', 'grey','brown'],
                  legend_labels=None,
                  cmap=None,
                  axis_one_label=None,
                  axis_two_label=None):

    axis_one_label=None if axis_one_label is not None else 'Principal Component 1'
    axis_two_label=None if axis_two_label is not None else 'Principal Component 2'

    labels_font_dict = {
        'family': 'monospace',
        'size': 20,
        'weight': 'bold',
        'style': 'italic',
        'color': 'black',
    }

    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = plt.gca()

    if color_mappings is None or len(color_mappings) == 0:
        color_mappings = np.arange(X_pca.shape[0])
        custom = False
        legend_labels = None
        print("No color_mappings provided — defaulting to sample index gradient.")
    
    unique_vals = np.unique(color_mappings)
    
    if custom:
        # Discrete category color mapping
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color_mappings, 
                             cmap=ListedColormap(colors_list[:len(unique_vals)]), alpha=0.6)
        
        if legend_labels is not None:
            legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=10, 
                                         markerfacecolor=color, label=label) 
                              for label, color in legend_labels.items()]
            ax.legend(handles=legend_handles, title="System Types", loc="upper right", prop={'size': 20, 'weight': 'bold'})

    else:
        # Use provided colormap, fallback to Greys
        norm = Normalize(vmin=np.min(color_mappings), vmax=np.max(color_mappings))
        cmap = cmap if cmap is not None else plt.get_cmap('Greys')

        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                             c=color_mappings, cmap=cmap, norm=norm, alpha=0.6)

        cbar_ticks = np.linspace(np.min(color_mappings), np.max(color_mappings), 10, dtype=int)
        
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            ticks=cbar_ticks, shrink=0.8, aspect=30, pad=0.02)
        
        cbar.set_label("Value", fontdict=labels_font_dict,
                       rotation=270, labelpad=25)
        
        cbar.ax.set_yticklabels([str(t) for t in cbar_ticks])

    # Final touches
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontdict=labels_font_dict)
    ax.set_xlabel(axis_one_label, fontdict=labels_font_dict)
    ax.set_ylabel(axis_two_label, fontdict=labels_font_dict)
    ax.tick_params(axis='x', colors='black')  
    ax.tick_params(axis='y', colors='black')

    plt.tight_layout()
    plt.savefig(savepath, dpi=500)
    plt.close()

def highlight_reps_in_embeddingspace(reduced_coordinates,
                    frame_list=((([80] * 20) + ([160] * 10)) * 2),
                    outfilepath='/zfshomes/lperez/thesis_figures/PCA/test_one_rep'):
    '''Visualizes and saves a replicates inside of embedding space

    Parameters
    ----------
    X_pca : np.ndarray, shape=(n_samples, n_components)
        The results of fitting a PCA analysis and using the .transform() method.

    frame_list : list of int, optional
        A list holding integer counts of the number of frames in each replicate. 
        Default is (([80] * 20) + ([160] * 10)) * 2.

    Returns
    -------
    None

    Notes
    -----
    Each replicate is plotted in its own subplot. A new row of plots begins every 30 replicates.
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm

    rep_iterator = 0
    
    for entry in range(len(frame_list)):

        colors = np.full(reduced_coordinates.shape[0], 'yellow')
        colors[rep_iterator:rep_iterator+frame_list[entry]] = 'blue'  

        # ticks for scaling
        x_min, x_max = reduced_coordinates[:, 0].min(), reduced_coordinates[:, 0].max()
        y_min, y_max = reduced_coordinates[:, 1].min(), reduced_coordinates[:, 1].max()
        plt.xticks(np.arange(np.floor(x_min), np.ceil(x_max) + 1, 1))
        plt.yticks(np.arange(np.floor(y_min), np.ceil(y_max) + 1, 1))

        plt.scatter(reduced_coordinates[:,0],reduced_coordinates[:,1],c=colors,s=5)
        plt.grid(visible=False)
        plt.savefig(f"{outfilepath}_rep{entry}.png")
        plt.close()
        
        rep_iterator+=frame_list[entry]
    
    return

#Contour plots 
def contour_embedding_space(outfile_path,embeddingspace_coordinates,levels=10,thresh=0,bw_adjust=.5):
    '''plot contour map of embedding space coordinates

    Paramters
    ---------
    embeddingspace_coordinates:arraylike,shape=(n_samples,2)
    This is the coordinates of your samples in the embedding space created
    by either UMAP or PCA. We assume you are reducing to two dimensions for 
    visualization purposes and therefore will use Gaussian KDE for estimation 
    via Seaborn.


    Returns
    -------
    None:
        Plots and saves


    Notes
    -----
    This just wraps the sns.kdeplot method for easy incorporation into our workflow and as such
    we provide essentially the same inputs for further customization you should go ahead and use
    the sns.kdeplot function yourself on the outputted matrices after reducing your original arrays. 


    
    Examples
    --------
    
    '''

    outfile_path = outfile_path if outfile_path is not None else os.getcwd()

    sns.kdeplot(
    x=embeddingspace_coordinates[:, 0],  # PC1
    y=embeddingspace_coordinates[:, 1],  # PC2
    fill=True,      # shaded contours
    cmap="cividis",
    levels=levels,
    thresh=thresh,#only plots regions where values are greater than some threshold 
    bw_adjust=bw_adjust,#wanted finer details which makes sense because we are looking for minority behaviors 
    cbar=True
    )

    plt.grid(visible=False)
    plt.savefig(outfile_path,dpi=800)
    
    return

if __name__ == "__main__":
    print('runnign just the visualization module')