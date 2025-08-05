import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import Normalize

#something to revisit is if this really needs any init or if it can just hold all functions in mem

class Systems_viz:

    def __init__(self):
        '''Holder for all of our visualizations

        Parameters
        ----------
        systems_analysis:systems_analysis,default=None
            An instance of a Systems Analysis object that was already previously created with the systems analysis
            class included 



        Returns
        -------
        none init 



        Notes
        -----




        Examples
        --------


        '''

        self.replicate_map_arr=None




    #Replicate maps
    def create_replicate_map_arr(self,labels,frame_list) -> np.ndarray:
        '''takes vector and reformats into array that can be visualized as timeseries with replicate map labels

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

        #some constants
        label_iterator=0
        reformatted_labels=[]
        max_frames = max(frame_list)

        #iterate through and just create values at every point in timeseries of replicates
        for rep_length in frame_list:
            current_replicate = np.copy(labels[label_iterator:label_iterator + rep_length]).astype(float)
            padded_replicate = np.pad(current_replicate, (0, max_frames - rep_length), constant_values=np.nan)
            reformatted_labels.append(padded_replicate)
            label_iterator+=rep_length

        #stack and return
        reformatted_labels = np.vstack(reformatted_labels)

        return reformatted_labels


    def traj_view_replicates(array, colors_list=['purple', 'orange', 'green', 'yellow', 'blue', 'red', 'pink', 'cyan', 'grey', 'brown'],
                                savepath='traj_view', title='Clusters per frame', xlabel='Frames', ylabel='Replicates',colormap=cm.magma_r):
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

        # Font dict here for global settings
        labels_font_dict = {
            'family': 'monospace',  # Font family (e.g., 'sans-serif', 'serif', 'monospace')
            'size': 20,             # Font size
            'weight': 'bold',       # Font weight ('normal', 'bold', 'light')
            'style': 'italic',      # Font style ('normal', 'italic', 'oblique')
            'color': 'black',       # Text color
        }
        
        norm = Normalize(vmin=np.nanmin(array), vmax=np.nanmax(array))
        cmap = colormap if colormap is not None else cm.plasma_r

        fig=plt.figure(figsize=(16,12)) 
        fig.tight_layout(pad=0) 

        #hold dictionary for values
        cluster_labels = {} 
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(array[i, j]):

                    x_pos = j 
                    y_pos = rows-i-1

                    color = cmap(norm(array[i, j]))  
                    cluster_label = 'Plasma Colormap'

                    scatter = plt.scatter(x_pos, y_pos,
                                        color=color,
                                        marker="P", 
                                        s=40)  


        unique_vals = np.unique(array[~np.isnan(array)]).astype(int)
        bounds = np.append(unique_vals, unique_vals[-1] + 1)  # to define edges between bins

        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(),
                        ticks=unique_vals, shrink=0.8, aspect=30, pad=0.02)

        cbar.set_label(" ", rotation=270, labelpad=10, fontsize=12, fontdict=labels_font_dict)
        cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black')
        cbar.ax.set_yticklabels([str(val) for val in unique_vals])
            
        #no border aesthetics
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        # Add labels, title, ticks, grid, then hide ticks and grid and show every 10
        plt.gca().set_xlabel(xlabel, fontdict=labels_font_dict)
        plt.gca().set_ylabel(ylabel, fontdict=labels_font_dict)
        plt.gca().set_title(title, fontdict=labels_font_dict)

        # Set tick positions based on original dimensions, but spaced out by the scaling factor
        plt.gca().set_xticks(np.arange(0, cols))  
        plt.gca().set_yticks(np.arange(0, rows))  
        plt.gca().set_xticklabels([str(i) if i % 80 == 0 else '' for i in range(cols)], fontdict=labels_font_dict)  
        plt.gca().set_yticklabels([str(i) if i % 30 == 0 else '' for i in range(rows-1, -1, -1)], fontdict=labels_font_dict)  

        plt.grid(False)
        plt.savefig(savepath, dpi=300)
        plt.close()

        return


#############
# Variables##
#############

replicate_frames = (([80] * 20) + ([160] * 10)) * 2

###############
# Example run
#####
GCU_frames=np.load('/Users/luis/Desktop/workspace/test_systems/redone_unrestrained_CCU_GCU_Trajectory_array.npy')
CGU_frames=np.load('/Users/luis/Desktop/workspace/test_systems/redone_unrestrained_CCU_CGU_Trajectory_array.npy')

print('frames loaded')
systems=[GCU_frames,CGU_frames]


from utilities.Analysis import systems_analysis 

n2_neighborhood = systems_analysis(systems)
print('systems object active')


test_instance = Systems_viz()
print('Visualization instance initiated')


X_pca,weights,explained_variance_ratio_ = n2_neighborhood.reduce_systems_representations()
filler_labels=weights[:,0]
test_instance.create_replicate_map_arr(filler_labels)
