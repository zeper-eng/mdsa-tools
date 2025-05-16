import numpy as np
import pandas as pd
import pycircos.pycircos as py
from t_a_Manipulation import create_dataframe_from_adjacency,filter_res
from Viz import make_MDCircos_object,base_mdcircos_graph
from matplotlib import cm

car_plusone_Asite_indexes=[94,127,240,408,409,410,423,424,425,426,427,428]

#---------------------------------------------------------------------------------------#
#Loading in our files and doing some dataframe manipulation upfront b/c thesis deadlines#
#---------------------------------------------------------------------------------------#

# we want our systems aggregated over the dimension of time 
redone_CCU_GCU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)
#print(redone_CCU_GCU_fulltraj.shape)
#print(redone_CCU_CGU_fulltraj.shape)
#print(redone_CCU_GCU_fulltraj[0,0,:])
#print(redone_CCU_CGU_fulltraj[0,0,:])
#print(redone_CCU_GCU_fulltraj[0,:,0])
#print(redone_CCU_CGU_fulltraj[0,:,0])



filtered_CCU_GCU_fulltraj = filter_res(redone_CCU_GCU_fulltraj,car_plusone_Asite_indexes)
filtered_CCU_CGU_fulltraj = filter_res(redone_CCU_CGU_fulltraj,car_plusone_Asite_indexes)

#print(filtered_CCU_GCU_fulltraj.shape,filtered_CCU_CGU_fulltraj.shape)
#print(filtered_CCU_GCU_fulltraj[0,0,:],filtered_CCU_CGU_fulltraj[0,0,:])
#print(filtered_CCU_GCU_fulltraj[0,:,0],filtered_CCU_CGU_fulltraj[0,:,0])





#quickly make dataframes with handy function I did indeed have on hand
CCU_GCU_dataframe, CCU_CGU_dataframe  = create_dataframe_from_adjacency(redone_CCU_GCU_fulltraj,'GCU'),create_dataframe_from_adjacency(redone_CCU_CGU_fulltraj,'CGU')
A_site_CCU_GCU_dataframe, A_site_CCU_CGU_dataframe  = create_dataframe_from_adjacency(filtered_CCU_GCU_fulltraj,'GCU'),create_dataframe_from_adjacency(filtered_CCU_CGU_fulltraj,'CGU')

#Merge dataframes full
final_df=pd.merge(CCU_GCU_dataframe,CCU_CGU_dataframe,on=["comparison"], how="inner")
final_df['abs_diff']=abs(final_df['GCU_Hbond']-final_df['CGU_Hbond'])
final_df['directional_diff_g_to_c']=final_df['GCU_Hbond']-final_df['CGU_Hbond']
final_df['circos_scaled_differences']=(final_df['abs_diff'] - np.min(final_df['abs_diff'])) / (np.max(final_df['abs_diff']) - np.min(final_df['abs_diff']))

#Merge dataframes A-site
final_Asite_df=pd.merge(A_site_CCU_GCU_dataframe,A_site_CCU_CGU_dataframe,on=["comparison"], how="inner")
final_Asite_df['abs_diff']=abs(final_Asite_df['GCU_Hbond']-final_Asite_df['CGU_Hbond'])
final_Asite_df['directional_diff_g_to_c']=final_Asite_df['GCU_Hbond']-final_Asite_df['CGU_Hbond']
final_Asite_df['circos_scaled_differences']=(final_Asite_df['abs_diff'] - np.min(final_Asite_df['abs_diff'])) / (np.max(final_Asite_df['abs_diff']) - np.min(final_Asite_df['abs_diff']))


#-----------------------------------------------#
#Creating a MDCircos plot and generating        #
#-----------------------------------------------#

#necessary inputs
residue_indexes = redone_CCU_GCU_fulltraj[0, 0, 1:].astype(int).tolist() #this can be anything im just in a rush so im pulling it out of the original matrix
A_site_indexes = filtered_CCU_GCU_fulltraj[0,0, 1:].astype(int).tolist()
diff_dict_all = final_df.set_index('comparison')['abs_diff'].to_dict()
diff_dict_A_site = final_Asite_df.set_index('comparison')['abs_diff'].to_dict()


#making ojects
Mdcircos_plt=make_MDCircos_object(residue_indexes)
Mdcircos_A_site_plt=make_MDCircos_object(A_site_indexes)

base_mdcircos_graph(Mdcircos_plt,residue_dict=diff_dict_all,savepath="/zfshomes/lperez/thesis_figures/Circos/CCU_abs_diff_Circos_",scale_factor=5,colormap=cm.cool_r)
base_mdcircos_graph(Mdcircos_A_site_plt,residue_dict=diff_dict_A_site,savepath="/zfshomes/lperez/thesis_figures/Circos/CCU_A_site_abs_diff_Circos",scale_factor=10,colormap=cm.cool_r)

#-----------------------------------------------#
#Creating a MDCircos plot with Directional diff #
#-----------------------------------------------#

#necessary inputs
residue_indexes = redone_CCU_GCU_fulltraj[0, 0, 1:].astype(int).tolist() #this can be anything im just in a rush so im pulling it out of the original matrix
A_site_indexes = filtered_CCU_GCU_fulltraj[0,0, 1:].astype(int).tolist()
diff_dict_all = final_df.set_index('comparison')['directional_diff_g_to_c'].to_dict()
diff_dict_A_site = final_Asite_df.set_index('comparison')['directional_diff_g_to_c'].to_dict()


#making ojects
Mdcircos_plt=make_MDCircos_object(residue_indexes)
Mdcircos_A_site_plt=make_MDCircos_object(A_site_indexes)

base_mdcircos_graph(Mdcircos_plt,residue_dict=diff_dict_all,savepath="/zfshomes/lperez/thesis_figures/Circos/CCU_directional_diff_Circos_",scale_factor=5,colormap=cm.cool_r)
base_mdcircos_graph(Mdcircos_A_site_plt,residue_dict=diff_dict_A_site,savepath="/zfshomes/lperez/thesis_figures/Circos/CCU_A_site_directional_diff_Circos",scale_factor=10,colormap=cm.cool_r)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


sorted_desc_df = final_df.sort_values('abs_diff', ascending=False)
sorted_desc_Asite_df = final_Asite_df.sort_values('abs_diff', ascending=False)


sorted_desc_df.to_csv('/zfshomes/lperez/final_thesis_data/circos_sorted_desc_df.csv')
sorted_desc_Asite_df.to_csv('/zfshomes/lperez/final_thesis_data/circos_sorted_desc_Asite_df.csv')


print(sorted_desc_df[['comparison','directional_diff_g_to_c','abs_diff']].head(40))
print(sorted_desc_Asite_df[['comparison','directional_diff_g_to_c','abs_diff']].head(40))

#-----------------------------------------------#
#PCA weights plots                              #
#-----------------------------------------------#
final_df=pd.read_csv('/zfshomes/lperez/final_thesis_data/all_pc1_squared_loadings.csv')
top_pc1=pd.read_csv('/zfshomes/lperez/final_thesis_data/top_pc1.csv')
top_pc2=pd.read_csv('/zfshomes/lperez/final_thesis_data/top_pc2.csv')

car_plusone_Asite_indexes=[94,127,240,408,409,410,423,424,425,426,427,428]
#necessary inputs
residue_indexes = redone_CCU_GCU_fulltraj[0, 0, 1:].astype(int).tolist() #this can be anything im just in a rush so im pulling it out of the original matrix
PC1_left_leaning, PC1_right_leaning = final_df[final_df['PC1']<0], final_df[final_df['PC1']>0]

print(len(PC1_left_leaning),len(PC1_right_leaning)) #46 and 57 add up to 
PC1_dict_left_leaning_dict,PC1_dict_right_leaning_dict = PC1_left_leaning.set_index('comparison')['PC1_squared'].to_dict(),PC1_right_leaning.set_index('comparison')['PC1_squared'].to_dict()
all_PC1_importantweights = final_df.set_index('comparison')['PC1_squared'].to_dict()

#making objects
PC1_left_plt,PC1_right_plt,pc1_all_plt=make_MDCircos_object(residue_indexes),make_MDCircos_object(residue_indexes),make_MDCircos_object(residue_indexes)
base_mdcircos_graph(PC1_left_plt,residue_dict=PC1_dict_left_leaning_dict,savepath="/zfshomes/lperez/thesis_figures/Circos/Grey_CCU_PC1_left_leaning_contributors_Circos_",colormap=cm.cool_r)
base_mdcircos_graph(PC1_right_plt,residue_dict=PC1_dict_right_leaning_dict,savepath="/zfshomes/lperez/thesis_figures/Circos/Grey_CCU_PC1_right_leaning_contributors_Circos_",colormap=cm.cool_r)




