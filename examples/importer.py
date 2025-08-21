import numpy as np
import mdtraj as md
from typing import Tuple, Dict
from mdsa_tools.Data_gen_hbond import cpptraj_hbond_import
import os

'''

First and foremost this is a an example of using our cpptraj import module. It doubles as a convenient
example of how to work with multiple files/trajectories/etc instead of using the concatenated workflow we suggest (although
a little hardcoded in terms of paths etc).

Part of the reality of working with MD trajectories is that very often you will be working with incredibly high volume
of data and as such the concatenated workflow is not always a reasonable use case.

'''

wt_files = [
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_WT_Short_HBondTime_Rep1.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_WT_Short_HBondTime_Rep2.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_WT_Short_HBondTime_Rep3.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_WT_Short_HBondTime_Rep4.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_WT_Short_HBondTime_Rep5.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_WT_Short_HBondTime_Rep6.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_WT_Short_HBondTime_Rep7.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_WT_Short_HBondTime_Rep8.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_WT_Short_HBondTime_Rep9.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_WT_Short_HBondTime_Rep10.dat",
]

y220c_pk11000_files = [
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_PK11000_Short_HBondTime_Rep1.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_PK11000_Short_HBondTime_Rep2.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_PK11000_Short_HBondTime_Rep3.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_PK11000_Short_HBondTime_Rep4.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_PK11000_Short_HBondTime_Rep5.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_PK11000_Short_HBondTime_Rep6.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_PK11000_Short_HBondTime_Rep7.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_PK11000_Short_HBondTime_Rep8.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_PK11000_Short_HBondTime_Rep9.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_PK11000_Short_HBondTime_Rep10.dat",
]

y220c_files = [
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_Short_HBondTime_Rep1.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_Short_HBondTime_Rep2.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_Short_HBondTime_Rep3.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_Short_HBondTime_Rep4.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_Short_HBondTime_Rep5.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_Short_HBondTime_Rep6.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_Short_HBondTime_Rep7.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_Short_HBondTime_Rep8.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_Short_HBondTime_Rep9.dat",
    "/zfshomes/sstetson/ShortVsLong/Analysis/HBond/p53_FL_Y220C_Short_HBondTime_Rep10.dat",
]

files=[wt_files,y220c_pk11000_files,y220c_files]

topologies = [
    '/zfshomes/sstetson/ShortVsLong/Trajectories/WT/Rep1/01_TLEAP/p53_WT_nowat.prmtop',
    '/zfshomes/sstetson/ShortVsLong/Trajectories/Y220C/Rep1/01_TLEAP/p53_Y220C_nowat.prmtop',
    '/zfshomes/sstetson/ShortVsLong/Trajectories/Y220C_PK11000/Rep1/01_TLEAP/p53_Y220C_PK11000_nowat.prmtop'
]

names=['wt','y220c_pk11000','y220c']


for i in range(len(files)):
    counter=1
    current_file_list=files[i]
    name=names[i]
    topology=topologies[i]

    for file in current_file_list:
        importer_instance=cpptraj_hbond_import(file,topology)
        rep=importer_instance.create_systems_rep()
        np.savez_compressed(f'/zfshomes/lperez/summer2025/workspace/compresserz/{name}_{counter}',rep=rep)
        counter+=1
