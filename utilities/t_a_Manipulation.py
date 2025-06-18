import numpy as np

def filter_res(array,residues_to_filter):
    """returns none 
        plots a given adjacency matrix and saves it as a png in the current directory
        
        Parameters
        ----------

        matrix:np.ndarray,shape=(n_frames,n_residues,n_frames)
                T_A_array, *need to add other description*
        
        name:str,default="adjacency_matrix"
            Name we would like to give to the matrix we are saving      
        
        axis_labels:list,default=None
            Default labelling we would like to use for our axis
            
        Returns
        -------
        None

        Examples
        --------

                
        Notes
        -----
        *since has been updated to also handle 3 dimensions just fine*
        The function assumes that you are providing adjacency matrices created with the rest of the pipeline
        so the function ignores the first row and column of the matrix assuming they are indexes.

        """
    #print(len(array.shape))
    residues_to_filter = [0]+residues_to_filter 
    
    if len(array.shape)==2:

        # Create a mask that marks the rows and columns to keep
        row_mask = np.isin(array[:, 0], residues_to_filter)
        col_mask = np.isin(array[0, :], residues_to_filter)

        filtered_rows=array[row_mask,:]
        filtered_array=filtered_rows[:,col_mask]
        #print(filtered_array.shape)
        
        
    #3dimensional filtering
    elif len(array.shape)==3:

        filtered_array=[]

        for i in range(array.shape[0]):

            current_frame = array[i,:,:]
            #print(f"current frame shape = {current_frame.shape}")
            #print("first loop")
            if len(current_frame.shape)==2:
                #print("second loop")
                #print(f"first row {current_frame[0,:]}")
                #print(f"first column {current_frame[:,0]}")
                row_mask = np.isin(current_frame[:, 0], residues_to_filter)
                col_mask = np.isin(current_frame[0, :], residues_to_filter)

                filtered_rows=current_frame[row_mask,:]
                filtered_frame=filtered_rows[:,col_mask]
                filtered_array.append(filtered_frame)

            else:
                print("frame not correctly indexed")
                break
            
        filtered_array=np.array(filtered_array)

    return filtered_array

def replicates_to_featurematrix(arrays)->np.ndarray:
    """returns an array formatted for kmeans clustering with scipy

    Parameters
    ----------
    arrays:list, expected=list[array_1,...,array_n-1] where each array is an np.ndarray with shape=(n_frames,n_residues,n_residues)
        Each array should have the shape as described above, where each frame has an adjacency matrix of pairwise
        comparisons between all residues. The only axis that can differ between arrays is the number of frames (n_frames).

        
    Returns
    -------
    array:np.ndarray,shape=(sum(n_frames),n_residues*n_residues) where the sum of n_frames refers to the total number of frames.
        Each row of the new matrix represents a flattened adjacency matrix for each frame, and the frames are stacked
        in such a way that each of the original arrays follow each other sequentially.


    Examples
    --------
    >>>CCU_GCU_fulltraj=np.load('/zfshomes/lperez/presentation_directory/CCU_GCU_Trajectory_array.npy',allow_pickle=True)
    >>>CCU_CGU_fulltraj=np.load('/zfshomes/lperez/presentation_directory/CCU_CGU_Trajectory_array.npy',allow_pickle=True)
    >>>arrays=[CCU_GCU_fulltraj,CCU_CGU_fulltraj]
    >>>Kmeans_replicatearray=format_replicate_for_clust(arrays)
        
        
    Notes
    -----
    First we concatenate all of our arrays into one large array. This does however rely on the premise that the only difference
    between our arrays is the number of frames (n_frames). This holds true conceptually because if we have a different
    number of pairwise residue comparisons we would not have comparable networks, or "systems".

    The goal is to flatten each adjacency matrix into a 1 dimensional vector and then stack all frames.
    This results in the expected formatting for scipy's kmeans clustering implementation where each .
    
    """
    
    #Concatenate arrays and define list to hold reformatted arrays
    concatenated_array=np.concatenate((arrays)) 
    final_frames=[]
    frame_num, n_residues, _ = concatenated_array.shape

    # Get indices for upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n_residues - 1, k=1)  # -1 due to [1:,1:] slice below


    final_frames = []
    for i in range(frame_num):
        current_frame = np.copy(concatenated_array[i, 1:, 1:])
        # Extract upper triangle only
        flattened = current_frame[triu_idx]
        final_frames.append(flattened)

    final_frames = np.vstack(final_frames).astype(np.float32)

    return final_frames

def get_labels_from_t_a(t_a_array,residues,mode="sum"):
    ''' returns a 1dimensional array of "average" values per entry in the array
    
    Parameters
    ----------
    t_a_array:np.ndarray,shape(n_frames,n_residues,n_residues)
        An averaged array (or individual frame of a trajectory) that is to be visualized.
        Since a typical adjacency matrix is NxN observations we will not focus on the multiple
        frames of the trajectory and will assume an averaged matrix has been provided. Alternatively
        this function can take a single frame from anywhere in the trajectory if you wish to analyze it

    residues:list,shape=(res_indexes,)
        A list denoting all of the residue indexes that you would like to analyze pairwise interactions for.
        This can be two residues meaning you just want to see the pairwise interactions between them or
        more residues and then you would get *all possible pairwise combinations of theese residues interactions*
    
    mode:string,default="sum",
        A string argument that decides the aggregation metric by which you would like to aggregate every frame. Mean of
        the residues of the CAR interaction surface for example, would give you the average hydrogen bonding found between all
        the residues of the car interaction surface and the +1 codon, sum would give you the total net hydrogen bond counts.
        
    Returns
    ----------
    avg_ta_labels:array,shape=(n_frames,):
        An array of the same size as the frames of interest except it just contains an average
        of all the possible pairwise hydrogen bonding interactions of the residues of interest for
        each frame
    
    Notes
    ----------
    This is actually a more powerful function than it may appear at first because if you only,
    say your pca found a few pairwise comparisons be incredibly important, well now we can isolate
    for just thoose frames and see which is best.

    '''

    # we always assume it comes with indixes but filter res accounts for this
    # this note in relation to the lines after filter res in the clauses

    if mode == "average":
        filtered_t_a_array=filter_res(t_a_array,residues)
        filtered_t_a_array=filtered_t_a_array[:,1:,1:]        
        ta_labels = np.mean(np.triu(filtered_t_a_array, k=1), axis=(1, 2)) #accounting for symmetry
    
    if mode == "sum":      
        filtered_t_a_array=filter_res(t_a_array,residues)
        filtered_t_a_array=filtered_t_a_array[:,1:,1:]
        ta_labels = np.sum(np.triu(filtered_t_a_array, k=1), axis=(1, 2)) #accounting for symmetry
    
    return ta_labels

#Circos
import pandas as pd
def prepare_t_a_for_abs_circos(trajectory,res_of_interest=None,keep_both=False):
    ''' prepares trajectory arrays for pairwise comparison absolute difference circos plot

    Parameters
    ----------
    trajectories:list, expected=list[array_1,...,array_n-1] where each array is an np.ndarray with shape=(n_frames,n_residues,n_residues)
        Each array should have the shape as described above, where each frame has an adjacency matrix of pairwise
        comparisons between all residues. The only axis that can differ between arrays is the number of frames (n_frames).

    res_of_interest:shape=(n_residues)
        Residues of interest here incase you would like to filter out specififc
        residues for the use of creating our circos plots
    
    keep_both:bool,default=False
        Really this was meant for me(Luis) but, in case you want the parameters for both the full trajectory you inputted
        and the filtered one the function will return both

    Returns
    -------
    attributes=tuple,shape=(n_trajectories+(res_interest*n_trajectories))
        A tuple with each of your aggregated trajectories as well as their filtered counterparts
        in sequential order a

    Examples
    --------


    Notes
    -----
    We filter the averaged matrix and if you have residues to filter append it immediately after the full trajectory
    in the list.
    
    So if you had:

    list = [a,b]

    then

    results = [average_a,average_a_filtered],[average_a_indexes,average_a_filtered_indexes]

    ''' 


    average_traj = np.mean(trajectory,axis=0)
    indexes=average_traj[0,:]


    #if we have residues to filter filter and also append to each list such that each sequential pair is of the same system
    if res_of_interest is not None:
        filtered_average_traj = filter_res(average_traj,res_of_interest) 

    

    return average_traj,indexes

# First lets define a function for just creating a dataframe from our 2x2 adjacency matrices

def create_dataframe_from_adjacency(array,sys_name,PCA_weights=None):
    '''returns a dataframe created from a given 2x2 adjacency matrix

    Parameters
    ----------
    array:np.ndarray
        An array of size nxn where n+1xn+1 is the number of residues. We assume the first row and column are indexes for future easy visualization
    
    Returns
    -------
    sorted_values_df:pd.Dataframe
        A pandas dataframe with the columns Residue 1, Residue 2, and Aggregated_Hbond_Value, where the index of each residue pair
        is used for the first ttwo columns respectively and then Average_hbond_value is their "average hydrogen bond count"

    Notes
    -----
        An important note is since this is made specifically for sparse matrices of residue comparisons, so in an effort to remove uneccessary entries and conserve working memory
        there are some data cleaning operations. 
        
        Namely:
        
        Residue-Residue comparisons on the diagonal are dropped. We assume theese are zero and since we are only interested in cross-residue comparisons they are of no importance to this project

        Residue1-Residue2 and Residue2-Residue1 comparisons are also simplified to just Res1-Res2 (a single entry), while the matrix being mirror symmetric is important for our network analysis
        and interpretation, they are not quite as important when gauging strictly values

        Residue1-Residue2 comparisons where the entry is 0 are also dropped because from the pespective of a dataframe we are really only interested in any comparisons where hydrogen bonds are present

        Theese turn out to be quite important ajdustments dropping our totals from about 52,000 rows to around 900

    Examples
    --------

    '''

    #extracting residue numbering, getting rid of index columns, and initializing final lists for pandas dict
    if len(array.shape) == 3:
        array=np.mean(array,axis=0)

    indexes=array[0,1:]
    print(indexes)
    array = array[1:,1:]
    comparison=[]

    #nested iteration setting up comparison lists and values list
    for i in range(len(indexes)):
        for j in range(len(indexes)):
            comparison.append([f"{int(indexes[i])}-{int(indexes[j])}", array[i, j]])

    
    #Creating dataframe and preforming all of the filters denoted in the docstring
    values_df=pd.DataFrame(comparison,columns=["comparison", f"{sys_name}_Hbond"])

    if PCA_weights is not None:
        values_df['pca_weight']=PCA_weights

    values_df["comparison"] = values_df["comparison"].apply(lambda x: "-".join(sorted(x.split("-"))))#
    values_df = values_df[values_df["comparison"].apply(lambda x: len(set(x.split("-"))) > 1)]
    values_df=values_df.drop_duplicates(subset=["comparison"])
    #sorted_values_df=values_df[values_df['aggregated_Hbond_Value']!=0]


    return values_df

def merge_pairwise_dataframes(df1,df2,):
    ''' merges a list containing pandas dataframes created with our pipeline

    Returns
    -------
    merged_df:Pandas.DataFrame,shape=()
        Merged pandas dataframe with absolute difference as new col
    

    Parameters
    ----------

    


    Examples
    --------

    


    Notes
    -----


    '''

    #create dataframe from values
    final_df=pd.merge(df1,df2,on=["comparison"], how="inner")
    final_df['abs_diff']=abs(final_df['GCU_Hbond']-final_df['CGU_Hbond'])
    final_df['directional_diff_g_to_c']=abs(final_df['GCU_Hbond']-final_df['CGU_Hbond'])
    final_df['circos_scaled_differences']=(final_df['abs_diff'] - np.min(final_df['abs_diff'])) / (np.max(final_df['abs_diff']) - np.min(final_df['abs_diff']))

    #check
    
    final_df.to_pickle("/zfshomes/lperez/final_thesis_data/master_database.pkl")

def extract_pairs_from_df(df,circos=False):
    '''' returns a list of integers containing indexes in comparisons column of a df

    Parameter
    ---------
    df:pd.Dataframe, 
        A pandas dataframe created from the create_dataframe_from_adjacency function

    Returns
    -------
    Comparisons:list,shape=(n_comparisons*2)
        Since the 

    Notes
    -----


    
    '''

    comparisons=df['comparison'].to_list()

    if circos ==True:#as tuples for our highlighting function
        comparisons = [item for sublist in [i.split('-') for i in comparisons] for item in sublist]
        final_comparisons=[]

        for i in range(0,len(comparisons),2):
            final_comparisons.append((comparisons[i],comparisons[i+1]))
        
        return final_comparisons

            

    comparisons = [int(item) for sublist in [i.split('-') for i in comparisons] for item in sublist]    

    return comparisons

