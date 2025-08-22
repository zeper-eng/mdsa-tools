from mdsa_tools.Analysis import systems_analysis
import numpy as np
import matplotlib.cm as cm
import os
import pandas as pd
from mdsa_tools.msm_modeler import MSM_Modeller as msm

#Pipeline setup assumed as in: Data Generation
redone_CCU_GCU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/Users/luis/Downloads/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)

from mdsa_tools.Viz import visualize_reduction
persys_frame_list=((([80] * 20) + ([160] * 10)))
persys_frame_short=([80] * 20) 
persys_frame_long= ([160] * 10)



#Just out of curiosity try just gcu
all_systems=[redone_CCU_GCU_fulltraj]
Systems_Analyzer = systems_analysis(systems_representations=all_systems,replicate_distribution=persys_frame_list)
Systems_Analyzer.replicates_to_featurematrix()
X_pca,_ ,_=Systems_Analyzer.reduce_systems_representations()
candidate_states_per_system=Systems_Analyzer.cluster_embeddingspace(outfile_path='./GCU_solo/GCU_pcaspace_clustersolo')
just_GCU_labels,centers=candidate_states_per_system[0]
visualize_reduction(X_pca,color_mappings=just_GCU_labels,savepath='./GCU_solo/GCU_pcaspace_clustersolo',cmap=cm.inferno_r)


#What if we tack thoose labels on to the full systems rep
all_systems=[redone_CCU_GCU_fulltraj,redone_CCU_CGU_fulltraj]
Systems_Analyzer = systems_analysis(systems_representations=all_systems,replicate_distribution=persys_frame_list)
Systems_Analyzer.replicates_to_featurematrix()
X_pca,_ ,_=Systems_Analyzer.reduce_systems_representations()
fourk_labels,fourk_centers=Systems_Analyzer.cluster_system_level(data=X_pca[0:3200,:],k=6)
GCU_with_filler=np.concatenate((fourk_labels,np.full(shape=(3200,),fill_value=np.max(fourk_labels)+1)))
visualize_reduction(X_pca,color_mappings=GCU_with_filler,savepath='./GCU_solo/fourk_GCU_on_original_pcaspace',cmap=cm.inferno)

perrep_frame_list=[]
for i in range(1,20+1):
    perrep_frame_list.append([i]*80)
for i in range(21,31):
    perrep_frame_list.append([i]*160)

perrep_frame_list=np.concatenate(perrep_frame_list)
print(f"perrep_frame_list:{perrep_frame_list.shape}")

fourk_labels,fourk_centers=Systems_Analyzer.cluster_system_level(data=X_pca[0:3200,:],k=6)
sixclust_filled=np.concatenate((fourk_labels,np.full(shape=(3200,),fill_value=np.max(fourk_labels)+1)))
perrep_with_filler=np.concatenate((perrep_frame_list,np.full(shape=(3200,),fill_value=np.max(perrep_frame_list)+1)))

visualize_reduction(X_pca,color_mappings=sixclust_filled,savepath='./GCU_solo/sixclust_GCU_on_original_pcaspace',cmap=cm.inferno)
visualize_reduction(X_pca,color_mappings=perrep_with_filler,savepath='./GCU_solo/perrep_GCU_on_original_pcaspace',cmap=cm.inferno)

UMAP=Systems_Analyzer.reduce_systems_representations(method='UMAP',n_neighbors=15,min_dist=.5)

visualize_reduction(UMAP,color_mappings=sixclust_filled,savepath='./GCU_solo/UMAP_sixk_GCU_on_original',cmap=cm.inferno)
visualize_reduction(UMAP,color_mappings=perrep_with_filler,savepath='./GCU_solo/UMAP_perrep_GCU_on_original',cmap=cm.inferno)

from mdsa_tools.Viz import highlight_reps_in_embeddingspace,highlight_crawl_directions,visualize_reduction
frame_list=((([80] * 20) + ([160] * 10)))
colormappings=[np.arange(0,np.max(i),1) for i in frame_list]
colormappings=np.concatenate(colormappings)

GCU_with_filler=np.concatenate((colormappings,np.full(shape=(3200,),fill_value=np.max(colormappings)+1)))
visualize_reduction(UMAP,color_mappings=GCU_with_filler,cmap=cm.magma_r,savepath='./GCU_solo/UMAP_sixclust_crawlspace',title='UMAP Dimensional Reduction with per-replicate frame highlighting',cbar_label='Frame Number')#all reps


###############################
#evaluating cohesion over time#
###############################
b = np.arange(3200) 
only_long  = (1600 <= b) 
only_short = (1600 > b) 

only_short_labels=fourk_labels[only_short]
only_long_labels=fourk_labels[only_long]

current_coordinates_short=X_pca[0:3200,:][only_short,:]
current_coordinates_long=X_pca[0:3200,:][only_long,:]

print(only_short_labels.shape)
print(only_long_labels.shape)

print(current_coordinates_short.shape)
print(current_coordinates_long.shape)


onlyshort_modeler=msm(only_short_labels,fourk_centers,current_coordinates_short,persys_frame_short)
onlylong_modeler=msm(only_long_labels,fourk_centers,current_coordinates_long,persys_frame_long)

onlyshort_results_shrinking=onlyshort_modeler.evaluate_cohesion_shrinkingwindow(step_size=20)
onlyshort_results_sliding=onlyshort_modeler.evaluate_cohesion_slidingwindow(step_size=20)

onlylong_results_shrinking=onlylong_modeler.evaluate_cohesion_shrinkingwindow(step_size=20)
onlylong_results_sliding=onlylong_modeler.evaluate_cohesion_slidingwindow(step_size=20)

onlyshort_results_shrinking.to_csv(f'GCU_solo_onlyshort_results_shrinking.csv')
onlyshort_results_sliding.to_csv(f'GCU_solo_onlyshort_results_sliding.csv')
onlylong_results_shrinking.to_csv(f'GCU_solo_onlylong_shrinkingresults.csv')
onlylong_results_sliding.to_csv(f'GCU_solo_onlylong_slidingresults.csv')


#################################################
#building replicate maps to visualize transition#
#################################################
from mdsa_tools.Viz import replicatemap_from_labels
print(fourk_labels.shape)
print(len(persys_frame_list))
GCU_with_filler=np.concatenate((fourk_labels,np.full(shape=(3200,),fill_value=np.max(fourk_labels)+1)))
replicatemap_from_labels(GCU_with_filler,persys_frame_list*2,savepath='./replicate_maps/6klust_replicate_map',title='6klust_replicate_map')
fourk_modeller=msm(fourk_labels,fourk_centers,X_pca[0:3200,:],frame_scale=persys_frame_list)
GCU_transition_prob_matrix = fourk_modeller.create_transition_probability_matrix()
np.savetxt('./GCU_solo/GCUsolo_transition_prob_matrix.csv',GCU_transition_prob_matrix,delimiter=',')
os._exit(0)

coordinates=[X_pca[0:3200,:],X_pca[3200:,:]]
