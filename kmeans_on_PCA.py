import numpy as np
import sys
sys.path.append("/zfshomes/lperez/final_thesis_scripts/pypure/utilities")
from t_a_Manipulation import replicates_to_featurematrix
from Clustering import preform_clust,preform_clust_opt
from Viz import label_iterator,traj_view_replicates_10by10
import os


#------------------------------------------
#Functions and paths necessary for this
#------------------------------------------
import sys
sys.path.append("/zfshomes/lperez/final_thesis_scripts/pypure/utilities/")
import numpy as np
from Convenience import test_list_2
from t_a_Manipulation import replicates_to_featurematrix
from Viz import create_2d_color_mappings,visualize_traj_PCA_onepanel
from dim_reduction import run_PCA

legend_labels=legend_labels = {
    'GCU Short': 'purple',
    'GCU Long (0-80)': 'orange',
    'GCU Long (80-160)': 'green',
    'CGU Short': 'yellow',
    'CGU Long (0-80)': 'blue',
    'CGU Long (80-160)': 'red'
}

#------------------------------------------------------------
#Loading in our files of interest and creating feature matrix
#------------------------------------------------------------

#load in our trajectories
redone_CCU_GCU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
print(redone_CCU_GCU_fulltraj.shape,redone_CCU_CGU_fulltraj.shape)
redone_arrays=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
redone_feature_matrix = replicates_to_featurematrix(arrays=redone_arrays)
X_pca,weights,explained_variance_ratio_=run_PCA(redone_feature_matrix,n=2)

#------------------------------------------------------------
#Loading in our files of interest and creating feature matrix
#------------------------------------------------------------
from Clustering import preform_clust_opt

replicate_frames = (([80] * 20) + ([160] * 10)) * 2

dimreduct_optimal_k_silhouette_labels,dimreduct_optimal_k_elbow_labels=preform_clust_opt(X_pca,outfile_path="/zfshomes/lperez/final_thesis_data/kluster_output/dimreduct_")


#------------------------------------------------------------
#Visualizing as PCA
#------------------------------------------------------------
visualize_traj_PCA_onepanel(X_pca,color_mappings=dimreduct_optimal_k_silhouette_labels,title="Sillouhete Labeled PCA of GCU and CGU Systems K="
                            ,savepath="/zfshomes/lperez/thesis_figures/PCA/dimreduct_sillouhete_colored_frames",clustering=False)

visualize_traj_PCA_onepanel(X_pca,color_mappings=dimreduct_optimal_k_elbow_labels,title="Elbow Labeled PCA of GCU and CGU Systems K=6"
                            ,savepath="/zfshomes/lperez/thesis_figures/PCA/dimreduct_elbow_colored_frames",clustering=False)

#------------------------------------------------------------
#Visualizing as Replicate Map
#------------------------------------------------------------
dimreduct_silhouette_viz_df=label_iterator(dimreduct_optimal_k_silhouette_labels,frame_list=replicate_frames)
dimreduct_optimal_k_elbow_labels = label_iterator(dimreduct_optimal_k_elbow_labels,frame_list=replicate_frames)

print(dimreduct_silhouette_viz_df.shape,dimreduct_optimal_k_elbow_labels.shape)

traj_view_replicates_10by10(dimreduct_silhouette_viz_df,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/dimreduct_optimal_silhouette_clust.png",clustering=False)
traj_view_replicates_10by10(dimreduct_optimal_k_elbow_labels,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/dimreduct_optimal_optimal_elbow_clust.png",clustering=False)
