
import numpy as np
import numpy.ma as ma
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colors import Normalize
from matplotlib import cm

''' 
A suite of functions meant to simplify the process of visualizing our data in new and more interpretable formats.

This is mainly built around a few inputs with example scripts found in the "test_cases.py" file

- A 1_dimensional vector of labels for each frame of your concatenated trajectory
-

'''

#Heatmaps
def plot_adjacency_matrix(matrix,name="adjacency_matrix",axis_labels=None,diff_matrix=False):
    import matplotlib.colors as mcolors
    """returns none 
        plots a given adjacency matrix and saves it as a png in the current directory
        
        Parameters
        ----------

        matrix:np.ndarray,shape=(1,n_residues,n_frames)
            An averaged array (or individual frame of a trajectory) that is to be visualized.
            Since a typical adjacency matrix is NxN observations we will not focus on the multiple
            frames of the trajectory and will assume an averaged matrix has been provided. Alternatively
            this function can take a single frame from anywhere in the trajectory if you wish to analyze it
     
        name:str,default="adjacency_matrix"
            Name we would like to give to the matrix we are saving      
        
        axis_labels:list,default=None
            Default labelling we would like to use for our axis
        
        diff_matrix:bool,default=False
            This is a simple boolean to evaluate whether the  matrix being provided is a difference matrix or not. 
            This is so the color scheme can change to a blue-orange scheme where blue is negative 0 is white and orange 
            is positive.
            
            
        Returns
        -------
        None

        Examples
        --------

                
        Notes
        -----
        The function assumes that you are providing adjacency matrices created with the rest of the pipeline
        so the function ignores the first row and column of the matrix assuming they are indexes.

        """


    # Create a figure, axes, and colorbar
    fig, ax = plt.subplots(figsize=(10, 8))
    if diff_matrix == False:
        im = ax.imshow(matrix, cmap='Oranges', aspect='auto', vmin=0, vmax=3)
    if diff_matrix == True:
        # Define custom colormap
        
        colors = [(0, 'blue'),  # Blue for negative values
                (0.5, 'white'),  # White for zero
                (1, 'orange')]  # Orange for positive values
        cmap = mcolors.LinearSegmentedColormap.from_list('orange_blue', colors)
        norm = mcolors.TwoSlopeNorm(vmin=np.min(matrix)-.01, vcenter=0, vmax=np.max(matrix)+.01) #small adjustment for testing 
        im = plt.imshow(matrix, cmap=cmap, interpolation='nearest', aspect='auto', norm=norm) 

    if axis_labels is not None:
        tick_positions = np.arange(matrix.shape[0])
        plt.xticks(rotation=45, ha='right',ticks=tick_positions,labels=axis_labels)
        plt.yticks(rotation=45, ha='right',ticks=tick_positions,labels=axis_labels)

    else:
        im = ax.imshow(matrix, cmap='Oranges', aspect='auto', vmin=0, vmax=3)
        plt.xticks(rotation=45, ha='right')
    
    cbar = ax.figure.colorbar(im, ax=ax, ticks=np.arange(0, 4))


    # Set labels and title
    ax.set_title(f'Heatmap of {name}')
    ax.set_xlabel('Residue Index')
    cbar.ax.set_ylabel('Residue Index', rotation=-90, va="bottom")



    # Save the figure
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    plt.show()

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

def traj_view_replicates(array,  colors_list=['purple', 'orange', 'green', 'yellow', 'blue', 'red', 'pink', 'cyan', 'grey','brown'],
                         clustering=True, savepath='traj_view', title='Clusters per frame', xlabel='Frames', ylabel='Replicates'):
    """Returns None

    A function for visualizing the clusters that all frames of each replicate end up in

    Parameters
    ----------
    array: np.ndarray shape=(n_replicates n_frames)
        Our final array where each row corresponds to a replicate
        and each column corresponds to a frame pertaining to that replicate
    
    colors_list: list-like default=['purple', 'orange', 'brown', 'green', 'yellow', 'blue', 'red', 'pink', 'cyan', 'grey']
        A list of colors to visualize clusters by default it contains 10 colors
        which is sufficient for clusters 0-9

    clustering: bool default=True
        Whether to visualize clusters using predefined colors from colors_list
        If False the function will use a colormap (e.g. viridis) based on normalized values
    
    savepath: str default='traj_view'
        The path where the plot will be saved
    
    title: str default='Clusters per frame'
        The title of the plot
    
    xlabel: str default='Frames'
        The label for the x-axis
    
    ylabel: str default='Replicates'
        The label for the y-axis

    Returns
    -------
    visualized_array: np.ndarray shape=(n_replicates*2 max_frames*2)
        A masked heatmap of the trajectory replicates returned for alternative use
    """

    rows, cols = array.shape

    # Use the Viridis colormap if clustering=False
    if not clustering:
        norm = plt.Normalize(vmin=np.nanmin(array), vmax=np.nanmax(array))
        cmap = cm.plasma_r
    
    # Create a dictionary to store cluster labels for the legend
    cluster_labels = {}

    for i in range(rows):
        for j in range(cols):
            if not np.isnan(array[i, j]):
                if clustering:
                    color_index = int(array[i, j]) % len(colors_list) 
                    color = colors_list[color_index]
                    cluster_label = f"Cluster {int(array[i, j])}"
                else:
                    color = cmap(norm(array[i, j]))  
                    cluster_label = 'Plasma colormap'

                scatter = plt.scatter(j, rows - i - 1,  # Mapping i to Matplotlib's row indexing
                                      color=color,
                                      marker='s', 
                                      s=1)

                # Add label to the dictionary (to make sure it's unique)
                if cluster_label not in cluster_labels:
                    cluster_labels[cluster_label] = scatter

    # Add legend for clustering, and colorbar for not clustering
    if clustering:
        plt.legend(cluster_labels.values(), cluster_labels.keys(), title="Clusters")
    
    if not clustering:
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca())
        cbar.set_label("Intensity (Plasma colormap)", rotation=270, labelpad=15)

    # Add labels, title, ticks, grid, then hide ticks and show every 30
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.gca().set_xticks(np.arange(0, cols, 1))
    plt.gca().set_yticks(np.arange(0, rows, 1))
    plt.grid(True, which='both', axis='both', linestyle='-', color='black', linewidth=0.05)

    plt.gca().set_xticklabels([str(i) if i % 30 == 0 else '' for i in np.arange(0, cols, 1)])
    plt.gca().set_yticklabels([str(i) if i % 30 == 0 else '' for i in np.flip(np.arange(0, rows, 1))])

    # Remove all borders (spines), save and display the plot
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    # Save the plot
    plt.xlim(-0.1, cols - 0.1)  # Tighten the margins
    plt.ylim(-0.1, rows - 0.1) 
    plt.savefig(savepath, dpi=300)
    plt.close()
    return

def traj_view_replicates_10by10_nogrid(array, colors_list=['purple', 'orange', 'green', 'yellow', 'blue', 'red', 'pink', 'cyan', 'grey', 'brown'],
                                clustering=True, savepath='traj_view', title='Clusters per frame', xlabel='Frames', ylabel='Replicates'):
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

    # Use the Viridis colormap if clustering=False
    if not clustering:
        norm = Normalize(vmin=np.nanmin(array), vmax=np.nanmax(array))
        cmap = cm.plasma_r 

    # Create a dictionary to store cluster labels for the legend
    cluster_labels = {}

    # Adjusting the scaling factor
    scaling_factor = 10  # Spacing factor for the grid

    for i in range(rows):
        for j in range(cols):
            if not np.isnan(array[i, j]):
                # Adjusting positioning to create the 10x10 space between squares
                x_pos = j * scaling_factor  # Scaling factor applied here
                y_pos = (rows - i - 1) * scaling_factor  # Adjust for Matplotlib's row inversion

                if clustering:
                    color_index = int(array[i, j]) % len(colors_list)
                    color = colors_list[color_index]
                    cluster_label = f"Cluster {int(array[i, j])}"
                else:
                    color = cmap(norm(array[i, j]))  
                    cluster_label = 'Plasma colormap'

                scatter = plt.scatter(x_pos, y_pos,
                                      color=color,
                                      marker='s', 
                                      s=1)  # Square size scaled

                # Add label to the dictionary (to make sure it's unique)
                if cluster_label not in cluster_labels:
                    cluster_labels[cluster_label] = scatter

    # Add legend for clustering, and colorbar for not clustering
    if clustering:
        plt.legend(cluster_labels.values(), cluster_labels.keys(), title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    if not clustering:
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca())
        cbar.set_label("Intensity (Plasma colormap)", rotation=270, labelpad=15)


    # Add labels, title, ticks, grid, then hide ticks and show every 10
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Set tick positions based on original dimensions, but spaced out by the scaling factor
    plt.gca().set_xticks(np.arange(0, cols * scaling_factor, scaling_factor))  # Adjust based on scaling factor
    plt.gca().set_yticks(np.arange(0, rows * scaling_factor, scaling_factor))  # Adjust based on scaling factor
    #plt.grid(True, which='both', axis='both', linestyle='-', color='black', linewidth=0.1)

    # Adjust tick labels for the correct spacing
    plt.gca().set_xticklabels([str(i) if i % 30 == 0 else '' for i in np.arange(0, cols, 1)])  # Label every 5th tick
    plt.gca().set_yticklabels([str(i) if i % 30 == 0 else '' for i in np.flip(np.arange(0, rows, 1))])  # Label every 5th tick

    # Remove all borders (spines), save and display the plot
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Tighten the margins to fit the 10x10 grid with spacing
    plt.ylim(-0.1, rows * scaling_factor - 0.1) 
    plt.xlim(-0.1, cols * scaling_factor - 0.1) 
    plt.savefig(savepath, dpi=300)
    plt.close()

    return

#fastest but less pretty
def traj_view_replicates_vectorized(array, colors_list=['purple', 'orange', 'green', 'blue', 'red', 'cyan', 'magenta', 'lime', 'teal', 'navy'],
                                clustering=True, savepath='traj_view', title='Clusters per frame', xlabel='Frames', ylabel='Replicates'):
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

   
    # Get the array shape
 
    # Create a new figure with black background
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('black')  # Set figure background to black
    ax.set_facecolor('black')  # Set axes background to black

    # Adjusting the scaling factor
    scaling_factor = 10  # Spacing factor for the grid

    # Create meshgrid for all positions
    x_indices, y_indices = np.meshgrid(np.arange(array.shape[1]), 
                                    np.arange(array.shape[0]))

    # Flatten the arrays for scatter plot
    x_indices = x_indices.flatten()

    # Fix the y_indices and flatten them correctly
    y_indices = (y_indices.flatten())[::-1]  # Reverse the y-indices and apply scaling

    values = array.flatten()
    
    if clustering:
        # Add label to the dictionary (to make sure it's unique)
        cluster_labels={f"Cluster {int(value)}":value for value in np.unique(values)}
        colors = [colors_list[int(i) % len(colors_list)] for i in values]
        ax.scatter(x_indices * scaling_factor, y_indices * scaling_factor,
                            color=colors,
                            marker="$f$", 
                            s=5)  # Keeping the "$f$" marker as requested        
        
        legend = ax.legend(cluster_labels.values(), cluster_labels.keys(), 
                          title="Clusters", bbox_to_anchor=(1.05, 1), 
                          loc='upper left', borderaxespad=0.)
        
        plt.setp(legend.get_title(), color='white')  # Set legend title color to white
        plt.setp(legend.get_texts(), color='white')  # Set legend text color to white
        legend.get_frame().set_facecolor('black')  # Set legend background to black
        legend.get_frame().set_edgecolor('white')  # Set legend edge color to white

    # Use the Viridis colormap if clustering=False
    if not clustering:
        norm = Normalize(vmin=np.nanmin(array), vmax=np.nanmax(array))
        cmap = cm.plasma
        color_spectrum = cmap(norm(values))  
        ax.scatter(x_indices, y_indices,
                            color=color_spectrum,
                            marker="$f$", 
                            s=5)  # Keeping the "$f$" marker as requested

        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label("Intensity (Plasma colormap)", rotation=270, labelpad=15, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    
    # Fix: Properly position ticks at the center of each grid cell
    x_ticks = np.arange(0, array.shape[1] * scaling_factor, scaling_factor)    
    y_ticks = np.arange(0, array.shape[0] * scaling_factor, scaling_factor)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
            
    # Add labels and title
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    ax.set_title(title, color='white')
            
    # Fix: Adjust tick labels without cutting off the first entries, and adjust limits
    ax.set_xticklabels([str(i) if i % 30 == 0 or i == 0 else '' for i in range(array.shape[1])], color='white')
    ax.set_yticklabels([str(i) if i % 30 == 0 or i == 0 else '' for i in range(array.shape[0]-1, -1, -1)], color='white')
    
    ax.set_xlim(-0.5, array.shape[1] - 0.5)
    ax.set_ylim(-0.5, array.shape[0] - 0.5)
    
    # Remove all borders (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

    return None

def traj_view_replicates_10by10(array, colors_list=['purple', 'orange', 'green', 'yellow', 'blue', 'red', 'pink', 'cyan', 'grey', 'brown'],
                                clustering=True, savepath='traj_view', title='Clusters per frame', xlabel='Frames', ylabel='Replicates',colormap=cm.magma_r):
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
        cmap = colormap if colormap is not None else cm.magma_r

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
        from matplotlib.colors import BoundaryNorm
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
import math
import pycircos.pycircos as py

def get_Circos_coordinates(residue, gcircle):
    """
    Return a 4-element tuple telling PyCircos chord_plot()
    to start in the middle of the arc with a radial anchor of 550.
    """
    arc = gcircle._garc_dict[residue]
    # The "size" is the arc length in PyCircos coordinates
    mid_position = arc.size * 0.5  # center of the arc
    # We'll anchor all chords at radial = 550
    # (this can be changed if your arcs are drawn in a different radial band)
    raxis_position = 550
    return (residue, mid_position, mid_position, raxis_position)

def make_MDCircos_object(residue_indexes):
    """
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

    if len(residue_indexes) < 50:
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


    if len(residue_indexes) >150:
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

def base_mdcircos_graph(empty_circle, residue_dict, savepath=os.getcwd()+'mdcircos_graph', scale_factor=5,colormap=cm.magma_r):
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

    Notes
    -----
    This is built as basically a wrapper for another python package so it is a little finicky in its implementation. In theory it should work fine
    with the other two functions and really only needs to be specific in the way that its taking the inputs for

    Examples
    --------

    '''
   
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import numpy as np

    # 0) Prep––your residue_dict, empty_circle, scale_factor, etc.

    # 1) Color normalization on the raw signed range
    vals = list(residue_dict.values())
    vmin, vmax = min(vals), max(vals)
    color_norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormap if colormap is not None else cm.plasma
    hex_color_map = {k: cmap(color_norm(v)) for k, v in residue_dict.items()}

    # 2) Width normalization on the absolute values via min–max
    abs_vals = [abs(v) for v in vals if v != 0]
    min_abs, max_abs = min(abs_vals), max(abs_vals)
    # avoid division by zero if all values are the same magnitude
    denom = max_abs - min_abs if max_abs != min_abs else 1.0

    width_norm = {
        k: (abs(v) - min_abs) / denom
        for k, v in residue_dict.items()
    }

    # 3) Plot chords
    fig, ax = plt.subplots(figsize=(6, 6))
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

                
def highlighted_mdcircos_graph(empty_circle, residue_dict_one,residue_dict_two, savepath=os.getcwd()+'highlighted_mdcircos_graph', scale_values=False):
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

    arcs_of_interest:list,shape(n_lists of pairwise tuples)

    Returns
    -------
    None

    Notes
    -----
    This is built as basically a wrapper for another python package so it is a little finicky in its implementation. In theory it should work fine
    with the other two functions and really only needs to be specific in the way that its taking the inputs for

    Examples
    --------

    '''

    import matplotlib

    plasma_cmap,viridis_cmap = cm.plasma_r,cm.viridis_r
    norm_plasma = Normalize(vmin=min(residue_dict_one.values()), vmax=max(residue_dict_one.values()))
    norm_viridis = Normalize(vmin=min(residue_dict_two.values()), vmax=max(residue_dict_two.values()))
    

    hex_color_map_plasma = {key: plasma_cmap(norm_plasma(value)) for key, value in residue_dict_one.items()}
    hex_color_map_viridis = {key: viridis_cmap(norm_viridis(value)) for key, value in residue_dict_two.items()}


    # Normalize the linewidth values if scale_values is True
    max_dictone_value,min_dictone_value =  max(residue_dict_one.values()),min(residue_dict_one.values())
    max_dicttwo_value,min_dicttwo_value = max(residue_dict_two.values()),min(residue_dict_two.values())
    
    for (key1, val1), (key2, val2) in zip(residue_dict_one.items(), residue_dict_two.items()):
    
        assert key1 == key2  # sanity check
    
        if val1 != 0:
            residue_one, residue_two = key1.split('-')
            arc_one = get_Circos_coordinates(residue_one, empty_circle)
            arc_two = get_Circos_coordinates(residue_two, empty_circle)
            
               # Normalize linewidth
            if scale_values is True:
                # Normalize the linewidth based on the min and max values in the residue_dict
                normalized_linewidth = (val1 - min_dictone_value) / (max_dictone_value - min_dictone_value) * 10  # Scaling factor of 10 for visualization
                color = hex_color_map_plasma[key1]
                empty_circle.chord_plot(arc_one, arc_two,
                                        linewidth=normalized_linewidth,
                                        facecolor=color,
                                        edgecolor=color)
                
            elif scale_values is False:
                empty_circle.chord_plot(arc_one, arc_two, linewidth=val1)

        
        if val2 != 0:
            residue_one, residue_two = key2.split('-')
            arc_one = get_Circos_coordinates(residue_one, empty_circle)
            arc_two = get_Circos_coordinates(residue_two, empty_circle)

               # Normalize linewidth
            if scale_values is True:
                # Normalize the linewidth based on the min and max values in the residue_dict
                normalized_linewidth = (val2 - min_dicttwo_value) / (max_dicttwo_value - min_dicttwo_value) * 10  # Scaling factor of 10 for visualization
                color = hex_color_map_viridis[key2]
                empty_circle.chord_plot(arc_one, arc_two,
                                        linewidth=normalized_linewidth,
                                        facecolor=color,
                                        edgecolor=color)
                
            elif scale_values is False:
                empty_circle.chord_plot(arc_one, arc_two, linewidth=val2)
    
        
    empty_circle.save(savepath,format="png",dpi=400)

def original_highlighted_mdcircos_graph(empty_circle, residue_dict_one,unique,unique_dos, savepath=os.getcwd()+'highlighted_mdcircos_graph', scale_values=False):
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

    arcs_of_interest:list,shape(n_lists of pairwise tuples)

    Returns
    -------
    None

    Notes
    -----
    This is built as basically a wrapper for another python package so it is a little finicky in its implementation. In theory it should work fine
    with the other two functions and really only needs to be specific in the way that its taking the inputs for

    Examples
    --------

    '''

    import matplotlib

    plasma_cmap,viridis_cmap = cm.plasma,cm.viridis
    norm = Normalize(vmin=min(residue_dict.values()), vmax=max(residue_dict.values()))

    hex_color_map_plasma = {key: plasma_cmap(norm(value)) for key, value in residue_dict.items()}
    hex_color_map_viridis = {key: viridis_cmap(norm(value)) for key, value in residue_dict.items()}


    # Normalize the linewidth values if scale_values is True
    min_value = min(residue_dict.values())
    max_value = max(residue_dict.values())
    
    for key, value in residue_dict.items(): 
        if value != 0:
            residue_one, residue_two = key.split('-')
            arc_one = get_Circos_coordinates(residue_one, empty_circle)
            arc_two = get_Circos_coordinates(residue_two, empty_circle)

            # Normalize the linewidth based on the min and max values in the residue_dict  # Scaling factor of 10 for visualization
            if (residue_one,residue_two) in unique: 
                 # Normalize the linewidth based on the min and max values in the residue_dict
                normalized_linewidth = (value - min_value) / (max_value - min_value) * 10  # Scaling factor of 10 for visualization
                color = hex_color_map_plasma[key]
                empty_circle.chord_plot(arc_one, arc_two,
                                        linewidth=normalized_linewidth,
                                        facecolor=color,
                                        edgecolor=color)
                
            elif scale_values is False:
                empty_circle.chord_plot(arc_one, arc_two, linewidth=value)
                

                
            if (residue_one,residue_two) in unique_dos: 
                color = '#ed7953'
                empty_circle.chord_plot(arc_one, arc_two,
                                    linewidth=value*10,
                                    facecolor=color,
                                    edgecolor=color)
    
    for key, value in residue_dict.items():
        if value != 0:
            residue_one, residue_two = key.split('-')
            arc_one = get_Circos_coordinates(residue_one, empty_circle)
            arc_two = get_Circos_coordinates(residue_two, empty_circle)

            # Normalize linewidth
            if scale_values is True:
                # Normalize the linewidth based on the min and max values in the residue_dict
                normalized_linewidth = (value - min_value) / (max_value - min_value) * 10  # Scaling factor of 10 for visualization
                color = hex_color_map[key]
                empty_circle.chord_plot(arc_one, arc_two,
                                        linewidth=normalized_linewidth,
                                        facecolor=color,
                                        edgecolor=color)
            elif scale_values is False:
                empty_circle.chord_plot(arc_one, arc_two, linewidth=value)
         
    empty_circle.save(savepath,format="png",dpi=400)



#Dimensional Reduction
import numpy as np
import matplotlib.pyplot as plt

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

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import numpy as np
import os

def visualize_traj_PCA_onepanel(X_pca, color_mappings, clustering=False, 
                                savepath=os.getcwd(), 
                                title="Principal Component Analysis (PCA) of GCU and CGU Systems", 
                                colors_list=['purple', 'orange', 'green', 'yellow', 'blue', 'red', 'pink', 'cyan', 'grey','brown'],
                                legend_labels = {'GCU Short': 'purple','GCU Long (0-80)': 'orange','GCU Long (80-160)': 'green','CGU Short': 'yellow','CGU Long (0-80)': 'blue','CGU Long (80-160)': 'red'},
                                cmap=cm.cool_r):
    ''' Visualizes data from an original feature matrix on two principal components after PCA

    Parameters
    ----------
    X_pca : np.ndarray, shape=(n_samples, n_components)
        The results of fitting a PCA analysis and using the .transform() method.

    color_mappings : list-like, shape=(n_samples)
        A list used to assign a color to each sample based on some mapping.
        If clustering=False, this should be a numeric array that will be mapped to a colormap.

    legend_labels : dict or None, default={'GCU Short': 'purple','GCU Long (0-80)': 'orange','GCU Long (80-160)': 'green','CGU Short': 'yellow','GCU Long (0-80)': 'blue','CGU Long (80-160)': 'red'}
        A dictionary mapping cluster labels to colors. If None, no legend is shown.

    clustering : bool, default=True
        If True, uses discrete colors from `colors_list`. If False, uses a colormap with a colorbar.

    savepath : str, default=current directory
        The full path where the output file will be saved.

    colors_list : list-like
        A list of colors to visualize discrete clusters.

    cmap : matplotlib colormap, default=cm.cool
        The continuous colormap to use when clustering=False.
    '''
    labels_font_dict = {
        'family': 'monospace',
        'size': 20,
        'weight': 'bold',
        'style': 'italic',
        'color': 'black',
    }

    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = plt.gca()

    unique_vals = np.unique(color_mappings)

    if clustering:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color_mappings, 
                             cmap=ListedColormap(colors_list[:len(unique_vals)]), alpha=0.6)

        if legend_labels is not None:
            legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=10, 
                                         markerfacecolor=color, label=label) 
                              for label, color in legend_labels.items()]
            ax.legend(handles=legend_handles, title="System Types", loc="upper right", prop={'size': 20, 'weight': 'bold'})

    else:
        # Ensure numeric values for boundaries
        unique_vals_numeric = np.unique([float(v) for v in color_mappings])
        boundaries = np.arange(min(unique_vals_numeric) - 0.5, max(unique_vals_numeric) + 1.5, 1)
        norm = BoundaryNorm(boundaries, cmap.N)

        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color_mappings, cmap=cmap, norm=norm, alpha=0.6)

        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            ticks=unique_vals_numeric, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label(label="Cluster Assignment", fontdict=labels_font_dict, rotation=270, labelpad=25)
        cbar.ax.yaxis.set_tick_params(color='black', labelsize=8)
        cbar.ax.set_yticklabels([str(int(val)) for val in unique_vals_numeric])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontdict=labels_font_dict)
    ax.set_xlabel("Principal Component 1", fontdict=labels_font_dict)
    ax.set_ylabel("Principal Component 2", fontdict=labels_font_dict)
    ax.tick_params(axis='x', colors='black')  
    ax.tick_params(axis='y', colors='black')

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()


def create_black_fig(x,y):
    '''' This is really just for the readibility of my own code to myself 

    Parameters
    ----------
    X:int,default=1
        number of rows of plots you want

    Y:int,default=1
        number of rows of plots you want

    Returns
    -------
    
    Examples
    --------

    Notes
    -----
    
    '''
    fig, ax = plt.subplots(x,y)
    # Create a new figure with black background
    plt.gca().set_facecolor('black')
    plt.gcf().patch.set_facecolor('black')  # Set figure background to black
    plt.gcf().set_facecolor('black')  # Set axes background to black

    if (x,y) != (1,1):
        for row in range(ax.shape[0]):
            for column in range(ax.shape[1]):
                current_ax=ax[row,column]
                current_ax.set_facecolor('black')

    return fig,ax

def create_PCA_on_rep(X_pca, frame_list=((([80] * 20) + ([160] * 10)) * 2)):
    '''
    Visualizes and saves a replicate mapping of embedded data.

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
    from matplotlib.colors import Normalize

    labels_font_dict = {
        'family': 'monospace',
        'size': 20,
        'weight': 'bold',
        'style': 'italic',
        'color': 'white',
    }

    cmap = cm.plasma
    rep_iterator = 0

    ''' # Create a figure with 2 rows and 30 columns
        fig, ax = create_black_fig(2, 30)
        fig.set_figheight(8)
        fig.set_figwidth(30)'''

    current_row_in_figures = 0

    for replicate in range(len(frame_list)):
        fig, ax = create_black_fig(1,1)

        current_rep_length = frame_list[replicate]

        if replicate % 30 == 0 and replicate != 0:
            current_row_in_figures += 1  # Move to next row every 30 replicates

        # Extract PCA data for this replicate
        current_replicate_labels = list(range(current_rep_length))
        current_replicate_X_PCA = X_pca[rep_iterator:rep_iterator + current_rep_length, 0].astype(float)
        current_replicate_Y_PCA = X_pca[rep_iterator:rep_iterator + current_rep_length, 1].astype(float)

        # Plot on the correct axis
        col_idx = replicate % 30
        current_ax = ax[current_row_in_figures, col_idx]

        current_rep_norm = Normalize(vmin=0, vmax=len(frame_list))
        current_ax.scatter(
            current_replicate_X_PCA,
            current_replicate_Y_PCA,
            c=current_replicate_labels,
            cmap=cmap,
            norm=current_rep_norm,
            alpha=0.6
        )

        current_ax.tick_params(axis='y', colors='white')

        rep_iterator += current_rep_length

    plt.savefig(f"/zfshomes/lperez/thesis_figures/PCA/test_one_rep{rep_iterator}.png")
    plt.close()

    return

def create_PCA_per_rep(X_pca,
                    frame_list=((([80] * 20) + ([160] * 10)) * 2),
                    name='/zfshomes/lperez/thesis_figures/PCA/test_one_rep'):
    '''
    Visualizes and saves a replicate mapping of embedded data.

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
    from matplotlib.colors import Normalize

    labels_font_dict = {
        'family': 'monospace',
        'size': 20,
        'weight': 'bold',
        'style': 'italic',
        'color': 'white',
    }

    cmap = cm.plasma
    rep_iterator = 0

    for replicate in range(len(frame_list)):
    
        fig, ax = create_black_fig(1,1)

        current_rep_length = frame_list[replicate]

        # Extract PCA data for this replicate
        current_replicate_labels = list(range(current_rep_length))
        current_replicate_X_PCA = X_pca[rep_iterator:rep_iterator + current_rep_length, 0].astype(float)
        current_replicate_Y_PCA = X_pca[rep_iterator:rep_iterator + current_rep_length, 1].astype(float)



        current_rep_norm = Normalize(vmin=0, vmax=len(frame_list))
        ax.scatter(
            current_replicate_X_PCA,
            current_replicate_Y_PCA,
            c=current_replicate_labels,
            cmap=cmap,
            norm=current_rep_norm,
            alpha=0.6
        )

        ax.tick_params(axis='y', colors='white')
        ax.set_yticks(np.arange(-8,8))
        ax.set_xticks(np.arange(-8,8))

        rep_iterator += current_rep_length

        plt.savefig(f"{name}{replicate}.png")
        plt.close()

     
        

    

    


    




