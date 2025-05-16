import numpy as np

GCU_not_processed = np.load("/zfshomes/lperez/datagen/CCU_GCU_estats_not_porcessed_3199.npy",allow_pickle=True)
CGU_not_processed = np.load("/zfshomes/lperez/datagen/CCU_CGU_estats_not_porcessed_3198.npy",allow_pickle=True)


with open('/zfshomes/lperez/final_thesis_scripts/data_files_created_from_scripts/not_processed_GCU','w') as outfile:
    for line in GCU_not_processed:
        outfile.write(print(line))
    outfile.close()
    
with open('/zfshomes/lperez/final_thesis_scripts/data_files_created_from_scripts/not_processed_CGU','w') as outfile:
    for line in GCU_not_processed:
        outfile.write(print(line))
    outfile.close()

