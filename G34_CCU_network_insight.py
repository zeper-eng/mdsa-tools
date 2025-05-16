import numpy as np
import pandas as pd
import networkx as nx
from utilities.network_viz import create_weighted_network, draw_colored_components,extract_measures,find_matching_residues,highlight_edges,create_weighted_highlighted_network

#Create graphs, extract measures
average_unrestrained_CCU_GCU_Trajectory_array=np.load("/zfshomes/lperez/final_thesis_data/average_unrestrained_CCU_GCU_Trajectory_array.npy",allow_pickle=True)
average_unrestrained_CCU_CGU_Trajectory_array=np.load("/zfshomes/lperez/final_thesis_data/average_unrestrained_CCU_CGU_Trajectory_array.npy",allow_pickle=True)
print(average_unrestrained_CCU_GCU_Trajectory_array.shape,average_unrestrained_CCU_CGU_Trajectory_array.shape)
GCU_network = create_weighted_network(average_unrestrained_CCU_GCU_Trajectory_array,"/zfshomes/lperez/thesis_figures/networks/GCU_avg_network.png")
CGU_network = create_weighted_network(average_unrestrained_CCU_CGU_Trajectory_array,"/zfshomes/lperez/thesis_figures/networks/CGU_avg_network.png")

Unique_to_GCU_Hbond= [[15, 16], [25, 428], [27, 95], [138, 27], [139, 27], [235, 27], [239, 27], [27, 353], [27, 465], [237, 28], [30, 31], [125, 49], [169, 49], [197, 51], [408, 68], [409, 68], [69, 72], [407, 69], [407, 72], [353, 91], [128, 93], [351, 93], [351, 94], [353, 95], [138, 96], [234, 97], [265, 97], [256, 99], [124, 169], [125, 482], [126, 171], [126, 242], [126, 484], [127, 169], [128, 408], [132, 134], [132, 137], [134, 137], [165, 174], [168, 174], [170, 421], [170, 423], [171, 409], [175, 425], [184, 464], [186, 383], [186, 425], [186, 426], [186, 427], [186, 465], [189, 412], [190, 424], [191, 412], [191, 420], [193, 423], [234, 241], [236, 263], [237, 480], [237, 487], [241, 242], [241, 428], [243, 487], [245, 488], [264, 266], [321, 324], [323, 379], [323, 411], [346, 347], [347, 349], [350, 374], [350, 375], [351, 353], [354, 382], [373, 374], [377, 425], [378, 425], [378, 426], [379, 412], [380, 384], [380, 427], [381, 383], [404, 411], [407, 414], [407, 423], [409, 422], [409, 426], [411, 424], [413, 422], [414, 421], [425, 464], [429, 485], [430, 480], [430, 489], [462, 467], [466, 468], [467, 468], [483, 487], [485, 487]]
Unique_to_CGU_Hbond=[[14, 16], [15, 427], [237, 24], [26, 27], [27, 357], [27, 381], [239, 29], [264, 29], [29, 427], [29, 428], [166, 46], [46, 483], [126, 48], [164, 49], [170, 51], [66, 72], [407, 68], [406, 69], [420, 69], [422, 69], [70, 72], [171, 70], [103, 71], [450, 71], [451, 71], [72, 75], [449, 72], [447, 76], [135, 88], [132, 90], [91, 95], [132, 91], [139, 92], [374, 92], [408, 92], [136, 93], [127, 95], [139, 95], [377, 95], [426, 95], [131, 96], [234, 96], [259, 97], [124, 98], [101, 124], [101, 451], [104, 484], [123, 125], [124, 233], [124, 269], [125, 485], [126, 427], [127, 168], [127, 408], [133, 258], [135, 138], [137, 140], [137, 258], [137, 264], [138, 261], [138, 351], [139, 350], [140, 262], [140, 351], [164, 170], [164, 197], [165, 172], [166, 193], [167, 428], [169, 426], [169, 428], [170, 172], [170, 194], [171, 422], [171, 426], [172, 192], [172, 425], [172, 426], [173, 186], [173, 423], [174, 185], [175, 185], [175, 189], [175, 193], [184, 461], [185, 187], [185, 289], [186, 379], [188, 411], [189, 379], [191, 197], [194, 198], [198, 421], [198, 422], [198, 423], [223, 237], [233, 244], [233, 269], [236, 427], [236, 429], [237, 427], [238, 242], [240, 426], [241, 427], [266, 268], [267, 269], [288, 321], [288, 322], [288, 323], [288, 325], [288, 411], [321, 401], [322, 383], [326, 375], [328, 374], [352, 353], [352, 380], [354, 358], [354, 426], [355, 357], [360, 362], [361, 362], [372, 374], [373, 375], [374, 378], [374, 408], [378, 382], [379, 382], [379, 383], [380, 426], [403, 412], [403, 415], [404, 406], [404, 412], [405, 413], [410, 413], [411, 421], [412, 415], [412, 420], [412, 421], [420, 423], [427, 487], [428, 482], [444, 451], [445, 446], [446, 449], [447, 448], [448, 449], [461, 467], [463, 464], [464, 465], [488, 489]]

GCU_Hbond_highlight = create_weighted_highlighted_network(average_unrestrained_CCU_GCU_Trajectory_array, residue_pairs_to_color=Unique_to_GCU_Hbond, outfile="/zfshomes/lperez/thesis_figures/networks/GCU_avg_network.png", color_components=False, electrostatics=False)
CGU_Hbond_highlight = create_weighted_highlighted_network(average_unrestrained_CCU_CGU_Trajectory_array, residue_pairs_to_color=Unique_to_CGU_Hbond, outfile="/zfshomes/lperez/thesis_figures/networks/CGU_avg_network.png", color_components=False, electrostatics=False)


import os
os._exit(0)
GCU_measuresdf=extract_measures(GCU_network)
CGU_measuresdf=extract_measures(CGU_network)


#Load in dataframe with pca weights and raw hydrogen bonding counts, extract top 20 contributing residues
master_dataframe = pd.read_pickle("/zfshomes/lperez/final_thesis_data/master_database.pkl")
pcasorted_master_dataframe=master_dataframe.sort_values(by="pca_weight",ascending=False)
top_ten_comparisons=pcasorted_master_dataframe['comparison'].head(10).tolist()
residues_in_important_pairing = [item for sublist in [i.split('-') for i in top_ten_comparisons] for item in sublist]

matching_dictionaries_GCU=find_matching_residues(GCU_measuresdf, residues_in_important_pairing, top_n=10)
matching_dictionaries_CGU=find_matching_residues(CGU_measuresdf, residues_in_important_pairing, top_n=10)


print(matching_dictionaries_GCU,'\n\n',matching_dictionaries_CGU)




GCU_networkcomp = draw_colored_components(GCU_network,"/zfshomes/lperez/thesis_figures/networks/GCU_avgcomp_network.png")
CGU_networkcomp = draw_colored_components(CGU_network,"/zfshomes/lperez/thesis_figures/networks/CGU_avgcomp_network.png")
