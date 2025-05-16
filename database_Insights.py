import numpy as np
import pandas as pd


master_dataframe = pd.read_pickle("/zfshomes/lperez/final_thesis_data/master_database.pkl")
pcasorted_master_dataframe=master_dataframe.sort_values(by="pca_weight",ascending=False)
print(pcasorted_master_dataframe.head(30))

I34_master_dataframe = pd.read_pickle("/zfshomes/lperez/final_thesis_data/I34_master_database.pkl")
I34_pcasorted_master_dataframe=I34_master_dataframe.sort_values(by="pca_weight",ascending=False)
print(I34_pcasorted_master_dataframe.head(30))



common_comparisons = set(pcasorted_master_dataframe.head(30)['comparison'].values) & set(I34_pcasorted_master_dataframe.head(30)['comparison'].values)

print(f"common comparisons in the top 10 for both include:\n\n{common_comparisons}")


