
from mdsa_tools.Data_gen_hbond import TrajectoryProcessor
from mdsa_tools.Analysis import systems_analysis
import numpy as np
import pandas as pd
import os

#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)


all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
Systems_Analyzer = systems_analysis(systems_representations=all_systems)
Systems_Analyzer.replicates_to_feature_matrix()

Global_UMAP_opt=Systems_Analyzer.reduce_systems_representations(method='UMAP',min_dist=.1,n_neighbors=15)
Local_UMAP_opt=Systems_Analyzer.reduce_systems_representations(method='UMAP',min_dist=1.0,n_neighbors=915)

from mdsa_tools.Viz import visualize_reduction


#basically a neat helper function for masking our systems the way we want to 
def make_replicate_ids(n400=20, n800=10, systems=2):
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

import colorcet as cc
replicate_palette = cc.glasbey[:60]  # list of hex colors


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
