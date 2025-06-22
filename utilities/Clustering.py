from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import typing
import numpy as np
import matplotlib.pyplot as plt

def format_replicate_for_clust(arrays)->np.ndarray:
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
    frame_num=concatenated_array.shape[0]
    
    #iterate through all frames and flatten, then stack into a 2d array.
    for i in range (0,frame_num):
        current_frame=np.copy(concatenated_array[i,:,:][1:,1:])
        flattened=current_frame.flatten() 
        final_frames.append(flattened)

    final_frames=np.vstack(final_frames).astype(np.float32)
    return final_frames

def preform_clust(array=None, n=None) -> np.ndarray:
    """Returns results of sklearn KMeans clustering implementation, including inertia.

    Parameters
    ----------
    array : numpy.ndarray, shape=(n_frames, n_residues*n_residues)
        Array formatted from previous operations where each row is a flattened adjacency matrix.
    n : int
        The number of initial centroids (clusters) for KMeans.

    Returns
    -------
    kmeans_output : np.ndarray, shape=(output,)
        An array holding all the relevant data from the KMeans clustering result:
        [cluster_centers_, inertia, labels_, distances_to_centroids]

    Examples
    --------
    >>> CCU_GCU_fulltraj = np.load('/path/to/CCU_GCU_Trajectory_array.npy', allow_pickle=True)
    >>> CCU_CGU_fulltraj = np.load('/path/to/CCU_CGU_Trajectory_array.npy', allow_pickle=True)
    >>> concatenated_array = join_cluster_systems([CCU_GCU_fulltraj, CCU_CGU_fulltraj])
    >>> Kmeans_replicatearray = format_replicate_for_clust(concatenated_array)
    >>> kluster_output = preform_clust(array=Kmeans_replicatearray, n=6)
    
    Notes
    -----
    The function uses the sklearn KMeans implementation and returns the cluster centers, inertia, labels, and distances.
    """

    # Initialize KMeans with the given number of clusters
    kmeans = KMeans(n_clusters=n)
    
    # Fit the model to the data
    kmeans.fit(array)
    
    # Get the cluster centers, inertia, and labels
    cluster_centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    labels = kmeans.labels_
    
    # Calculate the distances to the centroids for each sample
    distances_to_centroids = kmeans.transform(array)

    # Pack the results into a single numpy array
    kmeans_output = np.array([cluster_centers, inertia, labels, distances_to_centroids], dtype=object)
    
    return kmeans_output


# credit here to Jennifer Rose QAC 380 Ai Tools for data analysis
def preform_clust_opt(data,outfile_path, max_clusters=10):
    '''
    Parameters
    ----------
    data:np.ndarray,shape=(n_sample,n_features),
        A feature matrix of any kind, hopefully one provided from the rest of the pipeline but in theory, this is 
        just a scikit learn wrapper so you can plug anything you want really

    Returns
    ----------
    
    Notes
    ----------
    
    Examples
    ----------
    
    '''
    #keepinig track of our scores 
    inertia_scores,silhouette_scores,all_labels = [],[],[]
    cluster_range = range(2, max_clusters+2)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, init='random', n_init=k, random_state=0) #we set
        kmeans.fit(data) #fit data and now we have everything transformed
        cluster_centers, inertia, cluster_labels = kmeans.cluster_centers_,kmeans.inertia_,kmeans.labels_
        sil_score = silhouette_score(data, cluster_labels)
        

        inertia_scores.append(inertia)
        silhouette_scores.append(sil_score)
        all_labels.append(cluster_labels)


        np.save(f"{outfile_path}kluster_labels_{k}clust",cluster_labels)

    
    #so we save unless your calling this specific optimization
    from Viz import plot_elbow_scores,plot_sillohette_scores

    
    optimal_sillohuette=plot_sillohette_scores(cluster_range,silhouette_scores,outfile_path)
    optimal_elbow=plot_elbow_scores(cluster_range,inertia_scores,outfile_path)

    print(f"\nsize of labels:{len(all_labels)} ,optimal_elbow: {optimal_elbow}:optimal_sillohuette {optimal_sillohuette}")

    # Now you can return optimal k values
    
    optimal_k_silhouette_labels = all_labels[optimal_sillohuette-2] 
    optimal_k_elbow_labels = all_labels[optimal_elbow-2] 
    
    return optimal_k_silhouette_labels,optimal_k_elbow_labels

