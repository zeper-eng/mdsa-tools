import numpy as np
import os
from t_a_Manipulation import get_labels_from_t_a
from Viz import label_iterator,traj_view_replicates_10by10
from matplotlib import cm

replicate_frames=((([80]*20)+([160]*10))*2) 
CCU_GCU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
CCU_CGU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)

##########################################################################################################
##########################################################################################################
##########################################################################################################

#----------------------------------------------------------------------------------------------------
#Alternatively, it would be very cool to see what else we can find in regards to cool interactions
#----------------------------------------------------------------------------------------------------

#load in our trajectories and define care residues
P_residues_411 = [411,422]
P_residues_412 = [412,422]
CCAR_plusonen1 = [94,426]


#Filter for residues of interest, then average all the car residues so we have 1 value per frame of each replicate
foureleven_P_GCU,foureleven_P_CGU=get_labels_from_t_a(CCU_GCU_fulltraj,P_residues_411,mode="sum"),get_labels_from_t_a(CCU_CGU_fulltraj,P_residues_411,mode="sum")
fourtwelve_P_GCU,fourtwelve_P_CGU=get_labels_from_t_a(CCU_GCU_fulltraj,P_residues_412,mode="sum"),get_labels_from_t_a(CCU_CGU_fulltraj,P_residues_412,mode="sum")
CCAR_plusonen1_GCU,CCAR_plusonen1_CGU=get_labels_from_t_a(CCU_GCU_fulltraj,CCAR_plusonen1,mode="sum"),get_labels_from_t_a(CCU_CGU_fulltraj,CCAR_plusonen1,mode="sum")

foureleven_P_labels,fourtwelve_P_labels,ccarplusone_labels=np.concatenate([foureleven_P_GCU,foureleven_P_CGU]),np.concatenate([fourtwelve_P_GCU,fourtwelve_P_CGU]),np.concatenate([CCAR_plusonen1_GCU,CCAR_plusonen1_CGU])

#----------------------------------------------------------------------------------------------------
#Now we can do the same as in the visualization examples and simply take a look
#----------------------------------------------------------------------------------------------------


#replicate framelist as denoted in other examples, and savepath 
replicate_frames=((([80]*20)+([160]*10))*2) 
final_filename_411,final_filename_412="/zfshomes/lperez/thesis_figures/independent_analysis/Greys_P_site_411_Hbonding_per_frame","/zfshomes/lperez/thesis_figures/independent_analysis/Greys_P_site_412_Hbonding_per_frame"

#Use our exact same functions and apply a different colormap :)
reformatted_labels_411,reformatted_labels_412,ccar_plusone_reformatted =label_iterator(labels=foureleven_P_labels,frame_list=replicate_frames),label_iterator(labels=fourtwelve_P_labels,frame_list=replicate_frames),label_iterator(labels=ccarplusone_labels,frame_list=replicate_frames)

np.savetxt(f'/zfshomes/lperez/final_thesis_data/greys_411_422_Hbonding_per_frame.csv',reformatted_labels_411,delimiter=',')
np.savetxt(f'/zfshomes/lperez/final_thesis_data/greys_412_422_Hbonding_per_frame.csv',reformatted_labels_412,delimiter=',')
np.savetxt(f'/zfshomes/lperez/final_thesis_data/ccar_plusone_reformatted_Hbonding_per_frame.csv',ccar_plusone_reformatted,delimiter=',')

traj_view_replicates_10by10(reformatted_labels_411,title='greys_411-422_average_Hbonding_per_frame',savepath=final_filename_411,clustering=False,colormap=cm.cool_r) #note clustering is false
traj_view_replicates_10by10(reformatted_labels_412,title='greys_412-422_average_Hbonding_per_frame',savepath=final_filename_412,clustering=False,colormap=cm.cool_r) #note clustering is false



##########################################################################################################
##########################################################################################################
##########################################################################################################

#----------------------------------------------------------------------------------------------------
#Alternatively, it would be very cool to see what else we can find in regards to cool interactions
#----------------------------------------------------------------------------------------------------


#other large weightings are all generally not so close to the A-site
comparison_lists=[[412,422],[411,423],[410,424],[409,425],[408,426],[94,426]]
comparison_names=["411_423","410_424","409_425","408_426","408_426","94_426"]


#Filter for residues of interest, then average all the car residues so we have 1 value per frame of each replicate
results_GCU = list(map(lambda res_list: get_labels_from_t_a(CCU_GCU_fulltraj, res_list, mode="sum"), comparison_lists))
results_CGU = list(map(lambda res_list: get_labels_from_t_a(CCU_CGU_fulltraj, res_list, mode="sum"), comparison_lists))


concatenated_results=[]
for GCU_result,CGU_result in zip(results_GCU,results_CGU):
    current_result=np.concatenate((GCU_result,CGU_result))
    concatenated_results.append(current_result)


for i,result in enumerate(concatenated_results):
    reformatted_labels=label_iterator(labels=result,frame_list=replicate_frames)
    traj_view_replicates_10by10(reformatted_labels,title=f'{comparison_names[i]}_average_Hbonding_per_frame',
                                savepath="/zfshomes/lperez/thesis_figures/independent_analysis/Greys_"+f'{comparison_names[i]}_average_Hbonding_per_frame',clustering=False,colormap=cm.cool_r) #note clustering is false



os._exit(0)
##########################################################################################################
##########################################################################################################
##########################################################################################################


##########################################################################################################
##########################################################################################################
##########################################################################################################

#----------------------------------------------------------------------------------------------------
#Alternatively, it would be very cool to see what else we can find in regards to cool interactions
#----------------------------------------------------------------------------------------------------


#other large weightings are all generally not so close to the A-site
comparison_lists=[[167,168],[195,198],[354,380],[101,452],[164,48],[139,91]]
comparison_names=["167-168","195-198","354-380","101-452","164,48","139,91"]


#Filter for residues of interest, then average all the car residues so we have 1 value per frame of each replicate
results_GCU = list(map(lambda res_list: get_labels_from_t_a(CCU_GCU_fulltraj, res_list, mode="sum"), comparison_lists))
results_CGU = list(map(lambda res_list: get_labels_from_t_a(CCU_CGU_fulltraj, res_list, mode="sum"), comparison_lists))


concatenated_results=[]
for GCU_result,CGU_result in zip(results_GCU,results_CGU):
    current_result=np.concatenate((GCU_result,CGU_result))
    concatenated_results.append(current_result)


for i,result in enumerate(concatenated_results):
    reformatted_labels=label_iterator(labels=result,frame_list=replicate_frames)
    traj_view_replicates_10by10(reformatted_labels,title=f'{comparison_names[i]}_average_Hbonding_per_frame',
                                savepath="/zfshomes/lperez/thesis_figures/independent_analysis/"+f'{comparison_names[i]}_average_Hbonding_per_frame',clustering=False) #note clustering is false


os._exit(0)

#----------------------------------------------------------------------------------------------------
#Alternatively, it would be very cool to see what else we can find in regards to cool interactions
#----------------------------------------------------------------------------------------------------


#other large weightings are all generally not so close to the A-site
comparison_lists=[[167,168],[195,198],[354,380],[101,452]]
comparison_names=["167-168","195-198","354-380","101-452"]


#Filter for residues of interest, then average all the car residues so we have 1 value per frame of each replicate
results_GCU = list(map(lambda res_list: get_labels_from_t_a(CCU_GCU_fulltraj, res_list, mode="sum"), comparison_lists))
results_CGU = list(map(lambda res_list: get_labels_from_t_a(CCU_CGU_fulltraj, res_list, mode="sum"), comparison_lists))


concatenated_results=[]
for GCU_result,CGU_result in zip(results_GCU,results_CGU):
    current_result=np.concatenate((GCU_result,CGU_result))
    concatenated_results.append(current_result)


for i,result in enumerate(concatenated_results):
    reformatted_labels=label_iterator(labels=result,frame_list=replicate_frames)
    traj_view_replicates_10by10(reformatted_labels,title=f'{comparison_names[i]}_average_Hbonding_per_frame',
                                savepath="/zfshomes/lperez/thesis_figures/independent_analysis/"+f'{comparison_names[i]}_average_Hbonding_per_frame',clustering=False) #note clustering is false


os._exit(0)


##########################################################################################################
##########################################################################################################
##########################################################################################################

#----------------------------------------------------------------------------------------------------
#Alternatively, it would be very cool to see what else we can find in regards to cool interactions
#----------------------------------------------------------------------------------------------------

#load in our trajectories and define care residues
C_plusone1N1 = [94,426]
A_R_plusoneN2 = [240,422,427]


#Filter for residues of interest, then average all the car residues so we have 1 value per frame of each replicate
foureleven_P_GCU,foureleven_P_CGU=get_labels_from_t_a(CCU_GCU_fulltraj,C_plusone1N1,mode="sum"),get_labels_from_t_a(CCU_CGU_fulltraj,C_plusone1N1,mode="sum")
fourtwelve_P_GCU,fourtwelve_P_CGU=get_labels_from_t_a(CCU_GCU_fulltraj,A_R_plusoneN2,mode="sum"),get_labels_from_t_a(CCU_CGU_fulltraj,A_R_plusoneN2,mode="sum")

foureleven_P_labels,fourtwelve_P_labels=np.concatenate([foureleven_P_GCU,foureleven_P_CGU]),np.concatenate([fourtwelve_P_GCU,fourtwelve_P_CGU])

#----------------------------------------------------------------------------------------------------
#Now we can do the same as in the visualization examples and simply take a look
#----------------------------------------------------------------------------------------------------


#replicate framelist as denoted in other examples, and savepath 
replicate_frames=((([80]*20)+([160]*10))*2) 
final_filename_411,final_filename_412="/zfshomes/lperez/thesis_figures/independent_analysis/C_car_+1n1_Hbonding_per_frame","/zfshomes/lperez/thesis_figures/independent_analysis/A_R_car_+1n2_Hbonding_per_frame"

#Use our exact same functions and apply a different colormap :)
reformatted_labels_411,reformatted_labels_412=label_iterator(labels=foureleven_P_labels,frame_list=replicate_frames),label_iterator(labels=fourtwelve_P_labels,frame_list=replicate_frames)

traj_view_replicates_10by10(reformatted_labels_411,title='C_car_+1n1_Hbonding_per_frame',savepath=final_filename_411+'magmar',clustering=False,colormap=cm.magma_r) 
traj_view_replicates_10by10(reformatted_labels_412,title='A_R_car_+1n2_Hbonding_per_frame',savepath=final_filename_412+'magmar',clustering=False,colormap=cm.magma_r)

os._exit(0)

##########################################################################################################
##########################################################################################################
##########################################################################################################

#Filter for residues of interest, then average all the car residues so we have 1 value per frame of each replicate
avg_CAR_GCU=get_labels_from_t_a(CCU_GCU_fulltraj,CAR_residues,mode="sum")
avg_CAR_CGU=get_labels_from_t_a(CCU_CGU_fulltraj,CAR_residues,mode="sum")
avg_CAR_labels=np.concatenate([avg_CAR_GCU,avg_CAR_CGU])


#----------------------------------------------------------------------------------------------------
#Now we can do the same as in the visualization examples and simply take a look
#----------------------------------------------------------------------------------------------------


#replicate framelist as denoted in other examples, and savepath 
replicate_frames=((([80]*20)+([160]*10))*2) 
final_filename="/zfshomes/lperez/thesis_figures/independent_analysis/CAR_average_Hbonding_per_frame"

#Use our exact same functions and apply a different colormap :)
reformatted_CAR_vals=label_iterator(labels=avg_CAR_labels,frame_list=replicate_frames)

traj_view_replicates_10by10(reformatted_CAR_vals,title='CAR_average_Hbonding_per_frame',savepath=final_filename+'_no_grid',clustering=False) #note clustering is false


##########################################################################################################
##########################################################################################################
##########################################################################################################

#----------------------------------------------------------------------------------------------------
#Alternatively, it would be very cool to see what else we can find in regards to cool interactions
#----------------------------------------------------------------------------------------------------

PC1_80percent_contribution=[411, 422, 412, 422, 426, 94, 167, 168, 195, 198, 354, 380, 101, 452, 66, 75, 422, 423, 164, 48, 408, 425, 139, 91, 240, 427, 412, 413, 328, 403, 410, 411, 125, 126, 406, 413, 425, 94, 374, 380, 167, 172, 376, 426, 325, 412, 409, 424, 410, 423, 26, 467, 166, 167, 411, 412, 406, 412, 25, 30, 191, 424, 123, 488, 103, 104, 444, 75, 170, 195, 193, 197, 481, 488, 240, 426, 409, 410, 327, 374, 174, 192, 237, 429, 101, 126, 241, 428, 192, 193, 47, 48, 191, 423, 67, 72, 223, 429, 190, 191, 408, 94, 101, 125, 376, 380, 24, 466, 171, 425, 49, 50, 174, 191, 405, 68, 241, 427, 188, 288, 165, 193, 123, 124, 324, 410, 407, 412, 410, 424, 138, 139, 483, 486, 379, 409, 167, 482, 186, 464, 100, 101, 69, 70, 170, 48, 407, 409, 140, 350, 70, 71, 124, 125, 166, 172, 124, 488, 102, 125, 191, 192, 322, 325, 24, 25, 133, 98, 90, 91, 173, 424, 328, 402, 173, 192, 127, 426, 426, 427, 169, 170, 427, 428, 421, 51, 130, 448, 424, 425, 92, 95, 128, 171, 174, 425, 192, 194, 66, 67, 269, 99, 379, 425, 15, 27]
PC2_80percent_contribution=[411, 422, 409, 424, 66, 75, 167, 168, 376, 426, 406, 413, 410, 411, 425, 426, 328, 403, 424, 425, 324, 379, 166, 169, 124, 488, 240, 96, 191, 423, 384, 388, 240, 427, 379, 424, 234, 242, 168, 427, 190, 191, 412, 422, 174, 192, 444, 446, 169, 170, 67, 72, 101, 126, 127, 426, 127, 240, 70, 71, 240, 94, 103, 104, 411, 412, 237, 29, 237, 26, 124, 245, 169, 47, 354, 380, 191, 422, 327, 379, 486, 488, 408, 425, 480, 487, 128, 171, 422, 423, 174, 424, 131, 95, 407, 411, 101, 452, 429, 480, 69, 70, 127, 171, 193, 194, 423, 424, 126, 127, 450, 67, 173, 192, 134, 136, 171, 424, 195, 198, 130, 98, 376, 94, 408, 424, 324, 326, 444, 448, 169, 483, 126, 240, 134, 97, 238, 240]


#Filter for residues of interest, then average all the car residues so we have 1 value per frame of each replicate
PC1_80_labels_GCU=get_labels_from_t_a(CCU_GCU_fulltraj,PC1_80percent_contribution,mode="average")
PC1_80_labels_CGU=get_labels_from_t_a(CCU_CGU_fulltraj,PC1_80percent_contribution,mode="average")

Unique_labels=np.concatenate([PC1_80_labels_GCU,PC1_80_labels_CGU])

#----------------------------------------------------------------------------------------------------
#Now we can do the same as in the visualization examples and simply take a look
#----------------------------------------------------------------------------------------------------
from Viz import label_iterator,traj_view_replicates_10by10


#replicate framelist as denoted in other examples, and savepath 
replicate_frames=((([80]*20)+([160]*10))*2) 
final_filename="/zfshomes/lperez/thesis_figures/independent_analysis/Highest_contributing_PC1_Features"

#Use our exact same functions and apply a different colormap :)
reformatted_PC1_contributors=label_iterator(labels=Unique_labels,frame_list=replicate_frames)

traj_view_replicates_10by10(reformatted_PC1_contributors,title='PC1_80p_contributors_',savepath=final_filename+'_no_grid',clustering=False) #note clustering is false

##########################################################################################################
##########################################################################################################
##########################################################################################################

#----------------------------------------------------------------------------------------------------
#Alternatively, it would be very cool to see what else we can find in regards to cool interactions
#----------------------------------------------------------------------------------------------------

PC1_80percent_contribution_left = [412, 422, 426, 94, 167, 168, 195, 198, 354, 380, 101, 452, 164, 48, 139, 91, 240, 427, 410, 411, 406, 413, 374, 380, 167, 172, 376, 426, 411, 412, 25, 30, 191, 424, 103, 104, 193, 197, 409, 410, 327, 374, 174, 192, 241, 428, 67, 72, 190, 191, 49, 50, 174, 191, 123, 124, 410, 424, 138, 139, 483, 486, 100, 101, 70, 71, 102, 125, 191, 192, 24, 25, 133, 98, 328, 402, 169, 170, 427, 428, 130, 448, 424, 425, 92, 95, 174, 425, 379, 425, 15, 27]
PC1_80percent_contribution_right = [411, 422, 66, 75, 422, 423, 408, 425, 412, 413, 328, 403, 125, 126, 425, 94, 325, 412, 409, 424, 410, 423, 26, 467, 166, 167, 406, 412, 123, 488, 444, 75, 170, 195, 481, 488, 240, 426, 237, 429, 101, 126, 192, 193, 47, 48, 191, 423, 223, 429, 408, 94, 101, 125, 376, 380, 24, 466, 171, 425, 405, 68, 241, 427, 188, 288, 165, 193, 324, 410, 407, 412, 379, 409, 167, 482, 186, 464, 69, 70, 170, 48, 407, 409, 140, 350, 124, 125, 166, 172, 124, 488, 322, 325, 90, 91, 173, 424, 173, 192, 127, 426, 426, 427, 421, 51, 128, 171, 192, 194, 66, 67, 269, 99]

#Filter for residues of interest, then average all the car residues so we have 1 value per frame of each replicate
PC1_80_labels_GCU_left,PC1_80_labels_GCU_right=get_labels_from_t_a(CCU_GCU_fulltraj,PC1_80percent_contribution_left,mode="average"),get_labels_from_t_a(CCU_GCU_fulltraj,PC1_80percent_contribution_right,mode="average")
PC1_80_labels_CGU_left,PC1_80_labels_CGU_right=get_labels_from_t_a(CCU_CGU_fulltraj,PC1_80percent_contribution_left,mode="average"),get_labels_from_t_a(CCU_CGU_fulltraj,PC1_80percent_contribution_right,mode="average")

Unique_labels_left,Unique_labels_right=np.concatenate([PC1_80_labels_GCU_left,PC1_80_labels_CGU_left]),np.concatenate([PC1_80_labels_GCU_right,PC1_80_labels_CGU_right])

#----------------------------------------------------------------------------------------------------
#Now we can do the same as in the visualization examples and simply take a look
#----------------------------------------------------------------------------------------------------
from Viz import label_iterator,traj_view_replicates_10by10


#replicate framelist as denoted in other examples, and savepath 
replicate_frames=((([80]*20)+([160]*10))*2) 
final_filename_left,final_filename_right="/zfshomes/lperez/thesis_figures/independent_analysis/Highest_contributing_PC1_Features_left","/zfshomes/lperez/thesis_figures/independent_analysis/Highest_contributing_PC1_Features_right"

#Use our exact same functions and apply a different colormap :)
reformatted_PC1_contributors_left,reformatted_PC1_contributors_right=label_iterator(labels=Unique_labels_left,frame_list=replicate_frames),label_iterator(labels=Unique_labels_right,frame_list=replicate_frames)

traj_view_replicates_10by10(reformatted_PC1_contributors_left,title='PC1_80pct_left_contributors_',savepath=final_filename_left+'_no_grid',clustering=False) #note clustering is false
traj_view_replicates_10by10(reformatted_PC1_contributors_right,title='PC1_80pct_right_contributors_',savepath=final_filename_right+'_no_grid',clustering=False) #note clustering is false



##########################################################################################################
##########################################################################################################
##########################################################################################################

#----------------------------------------------------------------------------------------------------
#Alternatively, it would be very cool to see what else we can find in regards to cool interactions
#----------------------------------------------------------------------------------------------------

#load in our trajectories and define care residues
Unique_to_GCU_Hbond=[124, 169, 125, 482, 125, 49, 126, 171, 126, 242, 126, 484, 127, 169, 128, 408, 128, 93, 132, 134, 132, 137, 134, 137, 138, 27, 138, 96, 139, 27, 15, 16, 165, 174, 168, 174, 169, 49, 170, 421, 170, 423, 171, 409, 175, 425, 184, 464, 186, 383, 186, 425, 186, 426, 186, 427, 186, 465, 189, 412, 190, 424, 191, 412, 191, 420, 193, 423, 197, 51, 234, 241, 234, 97, 235, 27, 236, 263, 237, 28, 237, 480, 237, 487, 239, 27, 241, 242, 241, 428, 243, 487, 245, 488, 25, 428, 256, 99, 264, 266, 265, 97, 27, 353, 27, 465, 27, 95, 30, 31, 321, 324, 323, 379, 323, 411, 346, 347, 347, 349, 350, 374, 350, 375, 351, 353, 351, 93, 351, 94, 353, 91, 353, 95, 354, 382, 373, 374, 377, 425, 378, 425, 378, 426, 379, 412, 380, 384, 380, 427, 381, 383, 404, 411, 407, 414, 407, 423, 407, 69, 407, 72, 408, 68, 409, 422, 409, 426, 409, 68, 411, 424, 413, 422, 414, 421, 425, 464, 429, 485, 430, 480, 430, 489, 462, 467, 466, 468, 467, 468, 483, 487, 485, 487, 69, 72]
Unique_to_CGU_Hbond=[101, 124, 101, 451, 103, 71, 104, 484, 123, 125, 124, 233, 124, 269, 124, 98, 125, 485, 126, 427, 126, 48, 127, 168, 127, 408, 127, 95, 131, 96, 132, 90, 132, 91, 133, 258, 135, 138, 135, 88, 136, 93, 137, 140, 137, 258, 137, 264, 138, 261, 138, 351, 139, 350, 139, 92, 139, 95, 14, 16, 140, 262, 140, 351, 15, 427, 164, 170, 164, 197, 164, 49, 165, 172, 166, 193, 166, 46, 167, 428, 169, 426, 169, 428, 170, 172, 170, 194, 170, 51, 171, 422, 171, 426, 171, 70, 172, 192, 172, 425, 172, 426, 173, 186, 173, 423, 174, 185, 175, 185, 175, 189, 175, 193, 184, 461, 185, 187, 185, 289, 186, 379, 188, 411, 189, 379, 191, 197, 194, 198, 198, 421, 198, 422, 198, 423, 223, 237, 233, 244, 233, 269, 234, 96, 236, 427, 236, 429, 237, 24, 237, 427, 238, 242, 239, 29, 240, 426, 241, 427, 259, 97, 26, 27, 264, 29, 266, 268, 267, 269, 27, 357, 27, 381, 288, 321, 288, 322, 288, 323, 288, 325, 288, 411, 29, 427, 29, 428, 321, 401, 322, 383, 326, 375, 328, 374, 352, 353, 352, 380, 354, 358, 354, 426, 355, 357, 360, 362, 361, 362, 372, 374, 373, 375, 374, 378, 374, 408, 374, 92, 377, 95, 378, 382, 379, 382, 379, 383, 380, 426, 403, 412, 403, 415, 404, 406, 404, 412, 405, 413, 406, 69, 407, 68, 408, 92, 410, 413, 411, 421, 412, 415, 412, 420, 412, 421, 420, 423, 420, 69, 422, 69, 426, 95, 427, 487, 428, 482, 444, 451, 445, 446, 446, 449, 447, 448, 447, 76, 448, 449, 449, 72, 450, 71, 451, 71, 46, 483, 461, 467, 463, 464, 464, 465, 488, 489, 66, 72, 70, 72, 72, 75, 91, 95]


#Filter for residues of interest, then average all the car residues so we have 1 value per frame of each replicate
Unique_to_GCU_labels=get_labels_from_t_a(CCU_GCU_fulltraj,Unique_to_GCU_Hbond,mode="sum")
Unique_to_CGU_labels=get_labels_from_t_a(CCU_CGU_fulltraj,Unique_to_CGU_Hbond,mode="sum")

Unique_labels=np.concatenate([Unique_to_GCU_labels,Unique_to_CGU_labels])

#----------------------------------------------------------------------------------------------------
#Now we can do the same as in the visualization examples and simply take a look
#----------------------------------------------------------------------------------------------------
from Viz import label_iterator,traj_view_replicates_10by10


#replicate framelist as denoted in other examples, and savepath 
replicate_frames=((([80]*20)+([160]*10))) 

#Use our exact same functions and apply a different colormap :)
reformatted_GCU=label_iterator(labels=Unique_to_GCU_labels,frame_list=replicate_frames)
reformatted_CGU=label_iterator(labels=Unique_to_CGU_labels,frame_list=replicate_frames)


traj_view_replicates_10by10(reformatted_GCU,title='GCU_Unique_hbonds_total_Hbonding_per_frame',savepath="/zfshomes/lperez/thesis_figures/independent_analysis/GCU_Unique_total_Hbonding_per_frame"+'_no_grid',clustering=False) #note clustering is false
traj_view_replicates_10by10(reformatted_CGU,title='CGU_Unique_hbonds_total_Hbonding_per_frame',savepath="/zfshomes/lperez/thesis_figures/independent_analysis/CGU_Unique_total_Hbonding_per_frame"+'_no_grid',clustering=False) #note clustering is false

##########################################################################################################
##########################################################################################################
##########################################################################################################

#----------------------------------------------------------------------------------------------------
#Alternatively, it would be very cool to see what else we can find in regards to cool interactions
#----------------------------------------------------------------------------------------------------

PC1_80percent_contribution=[411, 422, 412, 422, 426, 94, 167, 168, 195, 198, 354, 380, 101, 452, 66, 75, 422, 423, 164, 48, 408, 425, 139, 91, 240, 427, 412, 413, 328, 403, 410, 411, 125, 126, 406, 413, 425, 94, 374, 380, 167, 172, 376, 426, 325, 412, 409, 424, 410, 423, 26, 467, 166, 167, 411, 412, 406, 412, 25, 30, 191, 424, 123, 488, 103, 104, 444, 75, 170, 195, 193, 197, 481, 488, 240, 426, 409, 410, 327, 374, 174, 192, 237, 429, 101, 126, 241, 428, 192, 193, 47, 48, 191, 423, 67, 72, 223, 429, 190, 191, 408, 94, 101, 125, 376, 380, 24, 466, 171, 425, 49, 50, 174, 191, 405, 68, 241, 427, 188, 288, 165, 193, 123, 124, 324, 410, 407, 412, 410, 424, 138, 139, 483, 486, 379, 409, 167, 482, 186, 464, 100, 101, 69, 70, 170, 48, 407, 409, 140, 350, 70, 71, 124, 125, 166, 172, 124, 488, 102, 125, 191, 192, 322, 325, 24, 25, 133, 98, 90, 91, 173, 424, 328, 402, 173, 192, 127, 426, 426, 427, 169, 170, 427, 428, 421, 51, 130, 448, 424, 425, 92, 95, 128, 171, 174, 425, 192, 194, 66, 67, 269, 99, 379, 425, 15, 27]
PC2_80percent_contribution=[411, 422, 409, 424, 66, 75, 167, 168, 376, 426, 406, 413, 410, 411, 425, 426, 328, 403, 424, 425, 324, 379, 166, 169, 124, 488, 240, 96, 191, 423, 384, 388, 240, 427, 379, 424, 234, 242, 168, 427, 190, 191, 412, 422, 174, 192, 444, 446, 169, 170, 67, 72, 101, 126, 127, 426, 127, 240, 70, 71, 240, 94, 103, 104, 411, 412, 237, 29, 237, 26, 124, 245, 169, 47, 354, 380, 191, 422, 327, 379, 486, 488, 408, 425, 480, 487, 128, 171, 422, 423, 174, 424, 131, 95, 407, 411, 101, 452, 429, 480, 69, 70, 127, 171, 193, 194, 423, 424, 126, 127, 450, 67, 173, 192, 134, 136, 171, 424, 195, 198, 130, 98, 376, 94, 408, 424, 324, 326, 444, 448, 169, 483, 126, 240, 134, 97, 238, 240]


#Filter for residues of interest, then average all the car residues so we have 1 value per frame of each replicate
PC1_80_labels_GCU=get_labels_from_t_a(CCU_GCU_fulltraj,PC1_80percent_contribution,mode="average")
PC1_80_labels_CGU=get_labels_from_t_a(CCU_CGU_fulltraj,PC1_80percent_contribution,mode="average")

Unique_labels=np.concatenate([PC1_80_labels_GCU,PC1_80_labels_CGU])

#----------------------------------------------------------------------------------------------------
#Now we can do the same as in the visualization examples and simply take a look
#----------------------------------------------------------------------------------------------------
from Viz import label_iterator,traj_view_replicates_10by10


#replicate framelist as denoted in other examples, and savepath 
replicate_frames=((([80]*20)+([160]*10))*2) 
final_filename="/zfshomes/lperez/thesis_figures/independent_analysis/Highest_contributing_PC1_Features"

#Use our exact same functions and apply a different colormap :)
reformatted_PC1_contributors=label_iterator(labels=Unique_labels,frame_list=replicate_frames)

traj_view_replicates_10by10(reformatted_PC1_contributors,title='PC1_80p_contributors_',savepath=final_filename+'_no_grid',clustering=False) #note clustering is false

##########################################################################################################
##########################################################################################################
##########################################################################################################

#----------------------------------------------------------------------------------------------------
#Alternatively, it would be very cool to see what else we can find in regards to cool interactions
#----------------------------------------------------------------------------------------------------

PC1_80percent_contribution_left = [412, 422, 426, 94, 167, 168, 195, 198, 354, 380, 101, 452, 164, 48, 139, 91, 240, 427, 410, 411, 406, 413, 374, 380, 167, 172, 376, 426, 411, 412, 25, 30, 191, 424, 103, 104, 193, 197, 409, 410, 327, 374, 174, 192, 241, 428, 67, 72, 190, 191, 49, 50, 174, 191, 123, 124, 410, 424, 138, 139, 483, 486, 100, 101, 70, 71, 102, 125, 191, 192, 24, 25, 133, 98, 328, 402, 169, 170, 427, 428, 130, 448, 424, 425, 92, 95, 174, 425, 379, 425, 15, 27]
PC1_80percent_contribution_right = [411, 422, 66, 75, 422, 423, 408, 425, 412, 413, 328, 403, 125, 126, 425, 94, 325, 412, 409, 424, 410, 423, 26, 467, 166, 167, 406, 412, 123, 488, 444, 75, 170, 195, 481, 488, 240, 426, 237, 429, 101, 126, 192, 193, 47, 48, 191, 423, 223, 429, 408, 94, 101, 125, 376, 380, 24, 466, 171, 425, 405, 68, 241, 427, 188, 288, 165, 193, 324, 410, 407, 412, 379, 409, 167, 482, 186, 464, 69, 70, 170, 48, 407, 409, 140, 350, 124, 125, 166, 172, 124, 488, 322, 325, 90, 91, 173, 424, 173, 192, 127, 426, 426, 427, 421, 51, 128, 171, 192, 194, 66, 67, 269, 99]

#Filter for residues of interest, then average all the car residues so we have 1 value per frame of each replicate
PC1_80_labels_GCU_left,PC1_80_labels_GCU_right=get_labels_from_t_a(CCU_GCU_fulltraj,PC1_80percent_contribution_left,mode="average"),get_labels_from_t_a(CCU_GCU_fulltraj,PC1_80percent_contribution_right,mode="average")
PC1_80_labels_CGU_left,PC1_80_labels_CGU_right=get_labels_from_t_a(CCU_CGU_fulltraj,PC1_80percent_contribution_left,mode="average"),get_labels_from_t_a(CCU_CGU_fulltraj,PC1_80percent_contribution_right,mode="average")

Unique_labels_left,Unique_labels_right=np.concatenate([PC1_80_labels_GCU_left,PC1_80_labels_CGU_left]),np.concatenate([PC1_80_labels_GCU_right,PC1_80_labels_CGU_right])

#----------------------------------------------------------------------------------------------------
#Now we can do the same as in the visualization examples and simply take a look
#----------------------------------------------------------------------------------------------------
from Viz import label_iterator,traj_view_replicates_10by10


#replicate framelist as denoted in other examples, and savepath 
replicate_frames=((([80]*20)+([160]*10))*2) 
final_filename_left,final_filename_right="/zfshomes/lperez/thesis_figures/independent_analysis/Highest_contributing_PC1_Features_left","/zfshomes/lperez/thesis_figures/independent_analysis/Highest_contributing_PC1_Features_right"

#Use our exact same functions and apply a different colormap :)
reformatted_PC1_contributors_left,reformatted_PC1_contributors_right=label_iterator(labels=Unique_labels_left,frame_list=replicate_frames),label_iterator(labels=Unique_labels_right,frame_list=replicate_frames)

traj_view_replicates_10by10(reformatted_PC1_contributors_left,title='PC1_80pct_left_contributors_',savepath=final_filename_left+'_no_grid',clustering=False) #note clustering is false
traj_view_replicates_10by10(reformatted_PC1_contributors_right,title='PC1_80pct_right_contributors_',savepath=final_filename_right+'_no_grid',clustering=False) #note clustering is false
