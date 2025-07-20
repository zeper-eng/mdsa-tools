import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import os
from sklearn.decomposition import PCA
from utilities.Viz import visualize_PCA
import pandas as pd
import matplotlib.cm as cm
import os

class systems_analysis:
    '''A big wrapper for conveniently storing a lot of our analysis methods
    '''

    def __init__(self,systems_representations):
        '''
        Parameters
        ----------
        systems_representations:list, expected=list[array_1,...,array_n-1] where each array is an np.ndarray with shape=(n_frames,n_residues,n_residues)
            Each array should have the shape as described above, where each frame has an adjacency matrix of pairwise
            comparisons between all residues. The only axis that can differ between arrays is the number of frames (n_frames).

        Returns
        -------
        None its an init

        Notes
        -----
        -automatically will convert list of arrays into a feature matrix so its easy to build it in
        -Additionally this for now expects systems of the same size for automation, if you want to run some of the downstream tasks
        you will most likely have to do it yourself if you are trying to analyze differently sized (in terms of frames) systems
        -Although since really this is a more general systems perspective on molecular dynamics its a great bouncing point for any systems project you may
        have in mind


        Examples
        --------
        '''
        
        self.num_systems=len(systems_representations) #this is useful later on for when we are doing system_specific operations
        self.systems_representations=systems_representations
        self.indexes = systems_representations[0][0,0,1:]
        self.feature_matrix=self.replicates_to_featurematrix(systems_representations)

        #this will be updated later and are defined within the functions themselves
        self.optimal_k_silhouette_labels=None
        self.optimal_k_elbow_labels=None
        self.pca_weights=None


        return

    #pre-processing
    def replicates_to_featurematrix(self,arrays)->np.ndarray:
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

    #Analyses
    def cluster_system_level(self,outfile_path, max_clusters=10,data=None):
        '''
        Parameters
        ----------
        data:np.ndarray,shape=(n_sample,n_features),
            A feature matrix of any kind, hopefully one provided from the rest of the pipeline but in theory, this is 
            just a scikit learn wrapper so you can plug anything you want really
        
        outfile_path:str,default=os.getcwd()
        

        Returns
        ----------
        
        Notes
        ----------
        
        Examples
        ----------
        
        '''
        outfile_path=outfile_path if outfile_path is not None else os.getcwd()
        optimal_k_silhouette_labels,optimal_k_elbow_labels,centers_sillohuette,centers_elbow=self.preform_clust_opt(outfile_path=outfile_path,data=self.feature_matrix)


        return optimal_k_silhouette_labels,optimal_k_elbow_labels,centers_sillohuette,centers_elbow
    
    def reduce_systems_representations(self,outfile_path=None,feature_matrix=None,n=None,colormappings=None,colormap=None,custom=None):
        '''
        Parameters
        ----------
        feature_matrix:np.ndarray,shape=(sum(n_frames),n_residues*n_residues) where the sum of n_frames refers to the total number of frames.
            Each row of the new matrix represents a flattened adjacency matrix for each frame, and the frames are stacked
            in such a way that each of the original arrays follow each other sequentially.
        
            
        n:int,default=2
            The number of principal components you would like to reduce your dataset down to
        
            
        str:outfile_path,default=os.getcwd()
            path to where you would like to save your visualization

        
        Returns
        -------
        X_pca,weights,explained_variance_ratio_

        As in the output from scikit learns module PCA() from the cluster.vq() module check in later for link 


        Notes
        -----
        You should include a pre-fix in your outfile path as the image will be saved with the ending
        "PCA_reduction" so a good example input is

        "/users/userone/desktop/project/output/test_"

        Examples
        --------

        
        '''
        outfile_path = outfile_path if outfile_path is not None else os.getcwd()
        feature_matrix = feature_matrix if feature_matrix is not None else self.feature_matrix
        n=n if n is not None else 2
        colormappings=colormappings if colormappings is not None else None
        colormap=colormap if colormap is not None else cm.cividis
        custom=custom if custom is not None else False
        
        X_pca,weights,explained_variance_ratio_=self.run_PCA(feature_matrix,n)
        visualize_PCA(X_pca,title="Sillouhete Labeled PCA of GCU and CGU Systems K=10",
                      color_mappings=colormappings,
                                savepath=f"{outfile_path}pca_reduction",custom=custom,cmap=colormap)
        return X_pca,weights,explained_variance_ratio_

    def cluster_embeddingspace(self,outfile_path=None,feature_matrix=None,n=None,max_clusters=None,elbow_or_sillohuette=None):
        '''A function for looking at conformational states in embedding space

        Parameters
        ----------
        outfilepath:str
            path to save

        featurematrix:np.Ndarray,default=self.featurematrix
            A feature matrix to be used for analysis
        
        n:int,default=2
            number of principal components to reduce to
        
        max_clusters:int,default=10
            The defualt number



        Returns
        -------



        Notes
        -----



        Examples
        --------

        '''

        outfile_path = outfile_path if outfile_path is not None else os.getcwd()
        feature_matrix = feature_matrix if feature_matrix is not None else self.feature_matrix
        n=n if n is not None else 2
        max_clusters=max_clusters if max_clusters is not None else 10
        elbow_or_sillohuette=elbow_or_sillohuette if elbow_or_sillohuette is not None else 'sillohuette'
        
        X_pca,weights,explained_variance_ratio_=self.run_PCA(feature_matrix,n)
        
        #grab the number of rows we need and then iterate through X_pca run kmeans and visualize using our initial values
        rows_per_system=X_pca.shape[0]/self.num_systems

        index=0


        for i in range(self.num_systems):
            # padding before and after so we can still visualize them all on the same scale
            before = index
            full_path=outfile_path+f"_lowdimclust_sys_{i+1}"
            current_system_frames = X_pca[index:index + int(rows_per_system), :]
           
            optimal_k_silhouette_labels,optimal_k_elbow_labels,centers_sillohuette,centers_elbow=self.preform_clust_opt(outfile_path=full_path,data=current_system_frames,max_clusters=max_clusters)
       
            if elbow_or_sillohuette == 'sillohuette':
                corelation_dataframe = self.create_pearsontest_for_kmeans_distributions(optimal_k_silhouette_labels,current_system_frames,centers_sillohuette)
                print(f"corelation_dataframe for system {i}\n{corelation_dataframe}")
                after = X_pca.shape[0] - (index + len(optimal_k_silhouette_labels))

                #this is where we use the previous spot for markings
                labels_filled = np.concatenate([
                np.full(before,(np.max(optimal_k_silhouette_labels)+1)),                      # padding before
                optimal_k_silhouette_labels,              # actual cluster labels
                np.full(after,(np.max(optimal_k_silhouette_labels)+1))                        # padding after
                ])

            elif elbow_or_sillohuette == 'elbow':
                corelation_dataframe = self.create_pearsontest_for_kmeans_distributions(optimal_k_elbow_labels,current_system_frames,centers_elbow)
                print(f"corelation_dataframe for system {i}\n{corelation_dataframe}")
                after = X_pca.shape[0] - (index + len(optimal_k_elbow_labels))

                #this is where we use the previous spot for markings
                labels_filled = np.concatenate([
                np.full(before, (np.max(optimal_k_elbow_labels)+1)),                      # padding before
                optimal_k_elbow_labels,              # actual cluster labels
                np.full(after, (np.max(optimal_k_elbow_labels)+1))                        # padding after
                ])

            print(len(labels_filled))

            visualize_PCA(X_pca,
                            color_mappings=labels_filled,
                            savepath=full_path,
                            title=f'optimal sillohuette clustering in embedding space for system {i}',
                            custom=False)
            index+=int(rows_per_system)
        
        self.pca_weights=weights

        return 
    
    def create_pearsontest_for_kmeans_distributions(self,labels,coordinates,cluster_centers):
        '''A function that is meant for the 

        Parameters
        ----------
        labels:listlike
            A list or array of labels that tell us which cluster each sample belongs to

        coordinates:array,shape=(n_samples,n_features)
            An array which tells us the coordinates of each sample so we can form distributions from them and run statistical tests
            (pearson corellation coefficient)
        
        cluster_centers:listlike,shape=k
            A list of arrays which tell us the coordinates for each cluster center so that we can calculate distributions
            to them


        Returns
        -------




        Notes
        -----



        Examples
        --------


        
        '''
        distances = np.linalg.norm(coordinates - cluster_centers[labels], axis=1) #euclidean distances to centroid

        #extracting everything by the group its a part of
        dist_by_cluster = {}
        for cluster_id in np.unique(labels):
            dist_by_cluster[int(cluster_id)] = distances[labels == cluster_id]

        #find minimum distance
        lengths = [len(i) for i in dist_by_cluster.values()]
        #print(lengths)
        shortest_length = min(lengths)
        #print(shortest_length)


        #form final list
        final_distributions=[]
        for i in dist_by_cluster.values():
            current_distribution=i[0:shortest_length,]
            final_distributions.append(current_distribution)
        
        from scipy.stats import pearsonr
        import pandas as pd
        
        correlations = []

        for i in range(len(final_distributions)):
            for j in range(i + 1,len(final_distributions)):
                r_value, p_value = pearsonr(final_distributions[i], final_distributions[j])
                correlations.append({
                    "cluster_i": i,
                    "cluster_j": j,
                    "pearson_r": r_value,
                    "p_value": p_value
                })

        correlation_df = pd.DataFrame(correlations)
        

        return correlation_df
       
    def create_PCA_ranked_weights(self, outfile_path=None, weights=None, indexes=None):
        '''A function for quickly creating the weights

        Parameters
        ----------



        Returns
        -------



        Examples
        --------



        Notes
        -----

        
        '''

        if weights is None:
            _,weights,_ =self.reduce_systems_representations
        if weights is not None:
            weights=weights

        outfile_path = outfile_path if outfile_path is not None else os.getcwd()
        indexes=indexes if indexes is not None else self.indexes

        # get shape info
        n_components, n_comparisons = weights.shape
        n_residues = self.systems_representations[0][0,1:,1:].shape[1] - 1  #grabbing original matrix  

        # grab only upper triangle
        triu_idx = np.triu_indices(len(indexes), k=1)

        # Generate comparison labels (no array values needed)
        comparisons = [f"{int(indexes[i])}-{int(indexes[j])}" for i, j in zip(*triu_idx)]
        dataframe={
            'Comparisons':comparisons,
            'PC1_Weights':weights[0],
            'PC2_Weights':weights[1],
            'PC1_magnitude':weights[0]**2,
            'PC2_magnitude':weights[1]**2,
            
        
        }

        dataframe=pd.DataFrame(dataframe).round(3)
        
        return dataframe

    def create_contour_plot(self,outfile_path=None,feature_matrix=None):
        '''
        Parameters
        ----------

        Notes
        -----

        Examples
        --------


        Returns
        -------
        
        '''
        feature_matrix=feature_matrix if feature_matrix is not None else self.feature_matrix
        outfile_path = outfile_path if outfile_path is not None else os.getcwd()
        
        X_pca,weights,explained_variance_ratio_=self.run_PCA(feature_matrix,2)
        
        import seaborn as sns
        import matplotlib.pyplot as plt



        '''
        Since we are not explicitly setting an h seaborn automatically picks one using scotts rule or silvermans rule
        the param would be bandwidth and can be adjusted using bw_adjust=.05 for instance creating narrower kernels

        bandwidth-> spread of the KDE (shape of the landscape)

        levels-> controls where to draw the contour lines (which KDE values)
            We arent setting levels so it can really be whatever at that point
            
        thresh-> hides regiosn below this KDE value (cut-off for noise)

        fill=true fills betweent he contours

        cbar = true


        '''
        sns.kdeplot(
            x=X_pca[:, 0],  # PC1
            y=X_pca[:, 1],  # PC2
            fill=True,      # shaded contours
            cmap="cviridis",
            thresh=0,#only plots regions where values are greater than some threshold 
            cbar=True
        )





        

        return

    #Algorithm wrappers 
    def preform_clust_opt(self,outfile_path, max_clusters=None,data=None):
        '''
        Parameters
        ----------
        data:np.ndarray,shape=(n_sample,n_features),
            A feature matrix of any kind, hopefully one provided from the rest of the pipeline but in theory, this is 
            just a scikit learn wrapper so you can plug anything you want really
        
        outfile_path:str,default=os.getcwd()
        

        Returns
        ----------
        
        Notes
        ----------
        
        Examples
        ----------
        
        '''
        data=data if data is not None else self.feature_matrix
        outfile_path = outfile_path if outfile_path is not None else os.getcwd()
        max_clusters = max_clusters if max_clusters is not None else 10
        
        #keeping track of our scores 
        inertia_scores,silhouette_scores,all_labels,centers = [],[],[],[]
        cluster_range = range(2, max_clusters+1)

        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, init='random', n_init=k, random_state=0) #we set
            kmeans.fit(data) #fit data and now we have everything transformed
            cluster_centers, inertia, cluster_labels = kmeans.cluster_centers_,kmeans.inertia_,kmeans.labels_
            sil_score = silhouette_score(data, cluster_labels)
            
            centers.append(cluster_centers)
            inertia_scores.append(inertia)
            silhouette_scores.append(sil_score)
            all_labels.append(cluster_labels)


            np.save(f"{outfile_path}kluster_labels_{k}clust",cluster_labels)

        
        #so we save unless your calling this specific optimization
        from utilities.Viz import plot_elbow_scores,plot_sillohette_scores

        
        optimal_sillohuette=plot_sillohette_scores(cluster_range,silhouette_scores,outfile_path)
        optimal_elbow=plot_elbow_scores(cluster_range,inertia_scores,outfile_path)

        #print(f"\nsize of labels:{len(all_labels)} ,optimal_elbow: {optimal_elbow}:optimal_sillohuette {optimal_sillohuette}")

        # Now you can return optimal k values
        
        optimal_k_silhouette_labels = all_labels[optimal_sillohuette-2] 
        optimal_k_elbow_labels = all_labels[optimal_elbow-2]
        centers_sillohuette = centers[optimal_sillohuette-2] 
        centers_elbow = centers[optimal_elbow-2] 

        
        return optimal_k_silhouette_labels,optimal_k_elbow_labels,centers_sillohuette,centers_elbow
    
    def run_PCA(self,feature_matrix,n):
        '''small function for running principal components analysis

        Parameters
        ----------

        feature_matrix:np.ndarray,shape=(sum(n_frames),n_residues*n_residues) where the sum of n_frames refers to the total number of frames.
            Each row of the new matrix represents a flattened adjacency matrix for each frame, and the frames are stacked
            in such a way that each of the original arrays follow each other sequentially.
        
        n:int,default=2
            The number of principal components you would like to reduce your dataset down to

        Returns
        -------




        Notes
        -----




        Examples
        --------



        '''

        pca=PCA(n_components=n)
        pca.fit(feature_matrix)
        new_values=pca.components_ 
        X_pca = pca.transform(feature_matrix)
        weights = pca.components_
        explained_variance_ratio_ = pca.explained_variance_ratio_

        print("X_pca shape (new data):",X_pca.shape)
        print(f"the total explained variance ratio is {np.sum(explained_variance_ratio_)}")
        print("weights shape:", weights.shape) 
        
        return X_pca,weights,explained_variance_ratio_