from sklearn.decomposition import PCA
import numpy as np

def run_PCA(feature_matrix,n):

    pca=PCA(n_components=n)
    pca.fit(feature_matrix)
    new_values=pca.components_ 
    X_pca = pca.transform(feature_matrix)
    weights = pca.components_
    explained_variance_ratio_ = pca.explained_variance_ratio_

    print("X_pca shape (new data):",X_pca.shape)
    print(f"the explained variance ratio is {explained_variance_ratio_}")
    print("weights shape:", weights.shape) 
    
    return X_pca,weights,explained_variance_ratio_

def peform_PCA_opt(feature_matrix,max_components=500):
    ''' Evaluates the best number of principle components for accounting for atleast 90% of the variance

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    Notes
    -----
    

    '''

    current_explained_variance=0.0
    i=2
    current_explained_variance=0

    if i <= max_components:
        while current_explained_variance<=.9:
            
            pca=PCA(n_components=i)
            pca.fit(feature_matrix)
            explained_variance_ratio_ = pca.explained_variance_ratio_
            total_variance_explained=np.sum(explained_variance_ratio_)
            current_explained_variance=total_variance_explained

    
    
    print(f"The ideal number of components to explain the variance: {current_explained_variance}")


    return 