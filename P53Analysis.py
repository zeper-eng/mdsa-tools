from utilities.Analysis import systems_analysis
import os
import numpy as np
from sklearn.decomposition import PCA

finalfullsystem_pk11000_system = np.load('/zfshomes/lperez/summer2025/workspace/test_systems/finalfullsystem_pk11000_system.npy')
finalfullsystem_wt_system = np.load('/zfshomes/lperez/summer2025/workspace/test_systems/finalfullsystem_wt_system.npy')
finalfullsystem_y220c_system = np.load('/zfshomes/lperez/summer2025/workspace/test_systems/finalfullsystem_y220c_system.npy')

all_systems=[finalfullsystem_pk11000_system,finalfullsystem_wt_system,finalfullsystem_y220c_system]
system_labels=[1]*2000+[2]*2000+[3]*2000 

#the goal of this is to analyze multiple differnt structures so naturally we need two different systems
Systems_Analyzer = systems_analysis(all_systems)
PCA_ranked_weights=Systems_Analyzer.create_PCA_ranked_weights()
top_pc1=PCA_ranked_weights.sort_values(by='PC1_magnitude', ascending=False)
print(top_pc1.head(10))
top_pc2=PCA_ranked_weights.sort_values(by='PC2_magnitude', ascending=False)
print(top_pc2.head(10))



os._exit(0)

#Clustering ju=t the systems representations of the trajectories we are interested in
optimal_k_silhouette_labels,optimal_k_elbow_labels,centers_sillohuette,centers_elbow = Systems_Analyzer.cluster_system_level(outfile_path='/zfshomes/lperez/summer2025/workspace/test_output/systems_kmeans/p53_',max_clusters=4)
print('clustering succesfully completed')
Systems_Analyzer.reduce_systems_representations(outfile_path='/zfshomes/lperez/summer2025/workspace/test_output/PCAp53_',colormappings=optimal_k_silhouette_labels)
Systems_Analyzer.reduce_systems_representations(outfile_path='/zfshomes/lperez/summer2025/workspace/test_output/PCAp53_',colormappings=system_labels)

print('PCA reduction succesful')
Systems_Analyzer.cluster_embeddingspace(outfile_path='/zfshomes/lperez/summer2025/workspace/test_output/cluster_embeddingspace/p53_',max_clusters=4,elbow_or_sillohuette='sillohuette')
print('Embedding space clustering succesfully completed')



##initital processing
'''
import os 
path='/zfshomes/lperez/summer2025/workspace/compresserz'
wt_files = [
    "/zfshomes/lperez/summer2025/workspace/compresserz/wt_1.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/wt_2.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/wt_3.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/wt_4.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/wt_5.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/wt_6.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/wt_7.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/wt_8.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/wt_9.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/wt_10.npz"
]
pk11000_files = [
    "/zfshomes/lperez/summer2025/workspace/compresserz/pk11000_1.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/pk11000_2.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/pk11000_3.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/pk11000_4.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/pk11000_5.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/pk11000_6.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/pk11000_7.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/pk11000_8.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/pk11000_9.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/pk11000_10.npz"
]
y220c_files = [
    "/zfshomes/lperez/summer2025/workspace/compresserz/y220c_1.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/y220c_2.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/y220c_3.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/y220c_4.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/y220c_5.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/y220c_6.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/y220c_7.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/y220c_8.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/y220c_9.npz",
    "/zfshomes/lperez/summer2025/workspace/compresserz/y220c_10.npz"
]


all_files=[wt_files,pk11000_files,y220c_files]
wt_system,pk11000_system,y220c_system=[],[],[]
all_systems=[wt_system,pk11000_system,y220c_system]

for i in range(len(all_files)):

    currrent_system = all_files[i]
    current_final_list = all_systems[i]

    for j in currrent_system:
        with np.load(j) as data:
            data = data["rep"]
            if 'y220c' in j:
                current_final_list.append(np.copy(data[::10,:-1,:-1]))
            if 'y220c' not in j:
                current_final_list.append(np.copy(data[::10,:,:]))
            del data
        
    all_systems[i] = np.concatenate(current_final_list, axis=0)

names=['wt_system','pk11000_system','y220c_system']

for i in range(len(all_systems)):
    np.save(f'/zfshomes/lperez/summer2025/workspace/test_systems/finalfullsystem_{names[i]}',all_systems[i])
'''