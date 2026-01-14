##################################################################################################################################################################
#Master file for all necessary data,and figures
#Excluding the structural visualizations (done in VMD and can be replicated easily)
#Anything else not seen here was masterfully done in trusty powerpoint (figs 1 and 2, organization of panels, parts of 5B etc)
##################################################################################################################################################################

##################################################################################################################################################################
# Necessary imports
##################################################################################################################################################################

from mdsa_tools.Analysis import systems_analysis
import numpy as np
import pandas as pd
from mdsa_tools.Data_gen_hbond import TrajectoryProcessor as tp
import os
from mdsa_tools.Convenience import unrestrained_residues
import matplotlib.pyplot as plt
from matplotlib import cm
import colorcet as cc
from mdsa_tools.Viz import visualize_reduction

##################################################################################################################################################################
# Original data generation
##################################################################################################################################################################


# paths (AMBER prmtop + mdcrd in this example)
system_one_topology = "5JUP_N2_GCU_nowat.prmtop"
system_one_trajectory = "redone_concatenated_GCU.mdcrd"

system_two_topology = "5JUP_N2_CGU_nowat.prmtop"
system_two_trajectory = "redone_concatenated_CGU.mdcrd"


# construct processors
traj_one = tp(trajectory_path=system_one_trajectory, topology_path=system_one_topology)
traj_two = tp(trajectory_path=system_two_trajectory, topology_path=system_two_topology)

# build per-frame adjacency matrices
system_one = traj_one.create_system_representations()
system_two = traj_two.create_system_representations()

print(system_one[0].shape)  # (n_res+1, n_res+1)
print(system_two[0].shape)

# (optional) focus on residues of interest
filtered_traj_one = traj_one.create_filtered_representations(residues_to_keep=unrestrained_residues)
filtered_traj_two = traj_two.create_filtered_representations(residues_to_keep=unrestrained_residues)

# save for later steps in the pipeline
outdir = os.getcwd()
os.makedirs(outdir, exist_ok=True)
np.save(os.path.join(outdir, "redone_unrestrained_CCU_GCU_Trajectory_array.npy"), system_one)
np.save(os.path.join(outdir, "redone_unrestrained_CCU_CGU_Trajectory_array.npy"), system_two)
np.save(os.path.join(outdir, "filtered_redone_unrestrained_CCU_GCU_Trajectory_array.npy"), filtered_traj_one)
np.save(os.path.join(outdir, "filtered_redone_unrestrained_CCU_CGU_Trajectory_array.npy"), filtered_traj_two)


##################################################################################################################################################################
#Global
#Visualization params
##################################################################################################################################################################


#Pipeline setup assumed as in: Data Generation:: or if you saved them yourself
redone_CCU_GCU_fulltraj=np.load('redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)


all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
Systems_Analyzer = systems_analysis(systems_representations=all_systems)
Systems_Analyzer.replicates_to_feature_matrix()

analyzer = systems_analysis([redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj])
analyzer.replicates_to_feature_matrix()

#preprocessing removes empty columns
mask=[]
filler_bool=False

for column in analyzer.feature_matrix.T:
    nonzero_col=np.where(column!=0)
    if nonzero_col[0].shape[0] > 0:
        mask.append(True)
        continue

    mask.append(False)

masked_feature_matrix = analyzer.feature_matrix[:,mask]
analyzer.feature_matrix = masked_feature_matrix #reassign to feature matrix


def make_replicate_ids(n400=20, n800=10, systems=2):
    '''basically a neat helper function for masking our systems the way we want to '''
    chunks = []
    cur = 1
    for _ in range(systems):
        chunks.append(np.repeat(np.arange(cur, cur + n400, dtype=np.int32), 80))
        cur += n400
        chunks.append(np.repeat(np.arange(cur, cur + n800, dtype=np.int32), 160))
        cur += n800
    return np.concatenate(chunks)

replicate_ids = make_replicate_ids()  # 60 uniques, length 6400


time_series_rep_lengths = (list(np.arange(0,80))*20 + list(np.arange(0,160))*10)

time_series_rep_lengths=time_series_rep_lengths*2

system_labels = 3200*[1] + 3200*[2] 

replicate_palette = cc.glasbey[:60]  # list of hex colors

##################################################################################################################################################################
#Kmeans 
#Clustering
#This also automatically saves the intermediate results and sillohuette plots to your savedirectory
#for more see the github repo
#(see end of doc for bargraphs)
##################################################################################################################################################################

k_labels_sil, k_labels_elbow, centers_sil, centers_elbow = Systems_Analyzer.perform_clust_opt(
    outfile_path=os.getcwd(),
    max_clusters=10
)


#For our particular analysis is assumed you end up with the 2 max clusters via sillohuette score evaluation thus, we simply take the differnece to compre
#ofcourse this would change slightly depending on your situation but that ad-hoc analyses would come down to the particular
#study that is being run

indexes=Systems_Analyzer.indexes
triu_idx = np.triu_indices(len(indexes), k=1)
comparisons = np.array([f"{str(int(indexes[i]))}-{str(int(indexes[j]))}" for i, j in zip(*triu_idx)])
centers_diff = centers_sil[0] - centers_sil[1]
print(centers_diff.shape,comparisons.shape)
difference_df=pd.DataFrame({"pairs":comparisons,
 "difference_in_feature_vals":centers_diff.round(3),
 "absolute_difference_in_feature_vals":abs(centers_diff).round(3)
 })

difference_df=difference_df.sort_values('absolute_difference_in_feature_vals',ascending=False).head(20)
difference_df.to_csv("difference_table_kmeans.csv")



##################################################################################################################################################################
#PCA
#Visualizations
##################################################################################################################################################################


X_pca,weights,explained_variance_ratio_ = Systems_Analyzer.reduce_systems_representations(
    outfile_path=os.getcwd(),         
    method="PCA"                         
)

print(f" our explained variance ratio is: {explained_variance_ratio_}")

visualize_reduction(X_pca,cbar_type='discrete',color_mappings=system_labels,savepath='PCA_1_in_50_global_system_labels',
title='1_in_50_system_labels',cmap=cm.plasma_r)

visualize_reduction(X_pca,cbar_type='discrete',color_mappings=time_series_rep_lengths,savepath='PCA_1_in_50_global_time_series_rep_lengths',
title='1_in_50_time_series_rep_lengths')

visualize_reduction(X_pca,cbar_type='discrete',color_mappings=replicate_ids,
color_palette=replicate_palette,savepath='PCA_1_in_50_global_replicate_palette',
title='1_in_50_replicate_palette')

##################################################################################################################################################################
#PCA
#Dataframes
##################################################################################################################################################################

PCA_ranked_weights = Systems_Analyzer.create_PCA_ranked_weights()  # DataFrame with PC loadings
PCA_ranked_weights.sort_values("PC1_magnitude", ascending=False).to_csv("CCU_G34_top_pc1.csv", index=False)
PCA_ranked_weights.sort_values("PC2_magnitude", ascending=False).to_csv("CCU_G34_top_pc2.csv", index=False)

##################################################################################################################################################################
#Circos
#Visualizations
##################################################################################################################################################################

outdir = os.getcwd()
os.makedirs(outdir, exist_ok=True)

from mdsa_tools.Viz import create_MDcircos_from_weightsdf

# Keep only the columns we need for filtering/plotting
PC1_df = PCA_ranked_weights[["Comparisons", "PC1_magnitude", "PC1_Weights"]]

pos = PC1_df[PC1_df["PC1_Weights"] > 0][["Comparisons", "PC1_magnitude"]]
neg = PC1_df[PC1_df["PC1_Weights"] < 0][["Comparisons", "PC1_magnitude"]]

create_MDcircos_from_weightsdf(pos, os.path.join(outdir, "PC1_positive"))
create_MDcircos_from_weightsdf(neg, os.path.join(outdir, "PC1_negative"))


##################################################################################################################################################################
#Replicate Map
#Visualizations
##################################################################################################################################################################


# Example pairs: (411, 422) and (412, 422)
hbond_counts_411_422 = Systems_Analyzer.extract_hbond_values(residues=[411, 422])
hbond_counts_412_422 = Systems_Analyzer.extract_hbond_values(residues=[412, 422])

frames = (([80] * 20) + ([160] * 10)) * 2


from mdsa_tools.Viz import replicatemap_from_labels
replicatemap_from_labels(
    labels=hbond_counts_411_422,
    frame_list=frames,
    savepath=os.path.join(outdir, "replicatemap_discrete_411_422")
)

replicatemap_from_labels(
    labels=hbond_counts_412_422,
    frame_list=frames,
    savepath=os.path.join(outdir, "replicatemap_discrete_412_422")
)


##################################################################################################################################################################
#Figuring out optimal UMAP parameters
##################################################################################################################################################################


perform_optimized_UMAP_global_df=Systems_Analyzer.perform_optimized_UMAP_global()
perform_optimized_UMAP_local_df=Systems_Analyzer.perform_optimized_UMAP_local()

print(perform_optimized_UMAP_global_df)
print(perform_optimized_UMAP_local_df)

from mdsa_tools.Viz import bubble_grid_manifoldlearning

bubble_grid_manifoldlearning(perform_optimized_UMAP_global_df)
bubble_grid_manifoldlearning(perform_optimized_UMAP_local_df)

##################################################################################################################################################################
#UMAP
#Visualizations
##################################################################################################################################################################


#First here are our reductions now that we know the optimal parameters for it(hardcoded for simplicity)
Global_UMAP_opt=Systems_Analyzer.reduce_systems_representations(method='UMAP',min_dist=.1,n_neighbors=15)
Local_UMAP_opt=Systems_Analyzer.reduce_systems_representations(method='UMAP',min_dist=1.0,n_neighbors=915)

#now onto visualizations
import matplotlib.cm as cm

visualize_reduction(Global_UMAP_opt,cbar_type='discrete',color_mappings=system_labels,savepath='1_in_50_global_system_labels',
title='1_in_50_system_labels',cmap=cm.plasma_r)

visualize_reduction(Global_UMAP_opt,cbar_type='discrete',color_mappings=time_series_rep_lengths,savepath='1_in_50_global_time_series_rep_lengths',
title='1_in_50_time_series_rep_lengths')

visualize_reduction(Global_UMAP_opt,cbar_type='discrete',color_mappings=replicate_ids,
color_palette=replicate_palette,savepath='1_in_50_global_replicate_palette',
title='1_in_50_replicate_palette')


visualize_reduction(Local_UMAP_opt,cbar_type='discrete',color_mappings=system_labels,savepath='1_in_50_local_system_labels',
title='1_in_50_system_labels',cmap=cm.plasma_r)

visualize_reduction(Local_UMAP_opt,cbar_type='discrete',color_mappings=time_series_rep_lengths,savepath='1_in_50_local_time_series_rep_lengths',
title='sa1_in_50_time_series_rep_lengths')

visualize_reduction(Local_UMAP_opt,cbar_type='discrete',
color_mappings=replicate_ids,savepath='1_in_50_local_replicate_palette',
title='1_in_50_replicate_palette',color_palette=replicate_palette)


##################################################################################################################################################################
#Non-package manuscript specific extras
##i.e. bargraphs
##################################################################################################################################################################

values=k_labels_sil.copy() #this is just adapting our previous stuff

values+=1

# Example input
labels = np.array(['CCU_GCU'] * 3200 + ['CCU_CGU'] * 3200)
values = np.array(values)  # Make sure it's a flat np.array of ints

# Get masks for each system
mask_gcu = labels == 'CCU_GCU'
mask_cgu = labels == 'CCU_CGU'

# Get value subsets
gcu_vals = values[mask_gcu]
cgu_vals = values[mask_cgu]

# Count unique values
gcu_unique, gcu_counts = np.unique(gcu_vals, return_counts=True)
cgu_unique, cgu_counts = np.unique(cgu_vals, return_counts=True)

# Build union of all value labels
all_labels = np.array(sorted(set(gcu_unique) | set(cgu_unique)))

# Align counts to full label set
def align_counts(unique_vals, counts, all_labels):
    label_to_count = dict(zip(unique_vals, counts))
    return np.array([label_to_count.get(k, 0) for k in all_labels])

gcu_aligned = align_counts(gcu_unique, gcu_counts, all_labels)
cgu_aligned = align_counts(cgu_unique, cgu_counts, all_labels)


bar_width = 0.4
x = np.arange(len(all_labels))

fig, ax = plt.subplots(figsize=(10, 6))


# a nice little work arround for grabbing the colors of the plasma colormap
from matplotlib import cm

plasma = cm.get_cmap('plasma')

# Use ends of the colormap spectrum
purple_color = plasma(0.0)   # start of plasma: dark purple
yellow_color = plasma(1.0)   # end of plasma: bright yellow

#this is where we actually add the bars
gcu_bars = ax.bar(x - bar_width/2, gcu_aligned, width=bar_width, label='CCU_GCU', color=purple_color)
cgu_bars = ax.bar(x + bar_width/2, cgu_aligned, width=bar_width, label='CCU_CGU', color=yellow_color)


# Add labels above bars
def add_bar_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 10,                    # spacing above bar
                str(height),
                ha='center', va='bottom',
                fontsize=20
            )

add_bar_labels(gcu_bars)
add_bar_labels(cgu_bars)

# setting the spines to false
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks(x)
ax.set_xticklabels(all_labels)
ax.set_xlabel('Cluster Assignment', fontsize=15, labelpad=15)
ax.set_ylabel('Number of Frames', fontsize=15, labelpad=15)
ax.set_title('Frame Counts per Cluster per System', fontsize=15, pad=15)

ax.legend(loc='center left', bbox_to_anchor=(1, 1), fontsize=12)
plt.tight_layout()
plt.savefig('framecluster_bars.png', dpi=500)
plt.show()
