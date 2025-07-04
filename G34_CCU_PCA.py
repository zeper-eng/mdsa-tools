#------------------------------------------
#Functions and paths necessary for this
#------------------------------------------
import sys
sys.path.append("/zfshomes/lperez/final_thesis_scripts/pypure/utilities/")
import numpy as np
from Convenience import test_list_2
from t_a_Manipulation import replicates_to_featurematrix
from Viz import create_2d_color_mappings,visualize_traj_PCA_onepanel,traj_view_replicates_10by10,label_iterator
from dim_reduction import run_PCA

legend_labels=legend_labels = {
    'GCU Short': 'purple',
    'GCU Long (0-80)': 'orange',
    'GCU Long (80-160)': 'green',
    'CGU Short': 'yellow',
    'CGU Long (0-80)': 'blue',
    'CGU Long (80-160)': 'red'
}

#super quick for me
replicate_frames = (([80] * 20) + ([160] * 10)) * 2
kmeans_labels_10k=np.load("/zfshomes/lperez/final_thesis_data/kluster_output/redone_kluster_labels_10clust.npy",allow_pickle=True)
kmeans_labels_10kdf=label_iterator(kmeans_labels_10k,frame_list=replicate_frames)
traj_view_replicates_10by10(kmeans_labels_10kdf,savepath=f"/zfshomes/lperez/thesis_figures/kmeans/redone_10k_clust.png",clustering=False)


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



#------------------------------------------
#Visualizing with the different sections of frames
#   Note: test_list_2 here is a 1d label vector like the kmeans output
#         where we are highlighting different frames of our system we would like to viz seperately
#         in this new 2d visualization
#
#------------------------------------------

#------------------------------------------
#Visualizing with the different sections of frames
#   Note: test_list_2 here is a 1d label vector like the kmeans output
#         where we are highlighting different frames of our system we would like to viz seperately
#         in this new 2d visualization
#
#------------------------------------------

colors_6kinds_of_frames = create_2d_color_mappings(labels=test_list_2,clustering=True)
visualize_traj_PCA_onepanel(
    X_pca,
    colors_6kinds_of_frames,
    legend_labels=legend_labels,
    title="Principal Component Analysis (PCA) of GCU and CGU Systems",
    savepath="/zfshomes/lperez/thesis_figures/PCA/redone_Onepanel_colored_frames",
    clustering=True
)

#------------------------------------------
#Visualizing with our labels from Kmeans_clustering
#   Note: It is assumed that the kmeans data would have been run in some capacity and saved for
#   an optimal number of K's so we will assume so and use it as our import
#
# Since we save every set of labels despite optimizing based on our evaluation metrics we can also pull out the 
# observations we are most interested in and also interested in comparing
#------------------------------------------

kmeans_labels_10k=np.load("/zfshomes/lperez/final_thesis_data/kluster_output/redone_kluster_labels_10clust.npy",allow_pickle=True).tolist()
visualize_traj_PCA_onepanel(X_pca,color_mappings=kmeans_labels_10k,title="Sillouhete Labeled PCA of GCU and CGU Systems K=10"
                            ,savepath="/zfshomes/lperez/thesis_figures/PCA/redone_sillouhete_colored_frames",clustering=False)

kmeans_labels_2k=np.load("/zfshomes/lperez/final_thesis_data/kluster_output/redone_kluster_labels_2clust.npy",allow_pickle=True).tolist()
visualize_traj_PCA_onepanel(X_pca,color_mappings=kmeans_labels_2k,title="Two Subsystems labeled PCA of GCU and CGU Systems K=2"
                            ,savepath="/zfshomes/lperez/thesis_figures/PCA/redone_distinct_systems_frames",clustering=False)

kmeans_8k_labels=np.load("/zfshomes/lperez/final_thesis_data/kluster_output/redone_kluster_labels_7clust.npy",allow_pickle=True).tolist()
visualize_traj_PCA_onepanel(X_pca,color_mappings=kmeans_8k_labels,title="Elbow Labeled PCA of GCU and CGU Systems K=7"
                            ,savepath="/zfshomes/lperez/thesis_figures/PCA/redone_elbow_colored_frames",clustering=False)
