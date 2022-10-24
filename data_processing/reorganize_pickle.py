# -*- coding: utf-8 -*-
"""reorganize_pickle.py

Re-organizes the dictionary in the rhythm-data.pickle file to 
create another dictionary that contains map of file names to the recording objects

Writes the ouput to a new pickle file: rhythm-data_file2object.pickle

@author: barisbozkurt
"""

import os
import shutil
import pickle
from datapreprocess_rhythm import Recording


# Reading and accessing features
data_file_name = '../data/rhythm/rhythm-data.pickle'
#reading pickle
with open(data_file_name, 'rb') as handle:
    rhy_data_read = pickle.load(handle)

path2object_dict = {}
for exercise in rhy_data_read.keys():
    for ref_rec in rhy_data_read[exercise]['ref']:
        path2object_dict[ref_rec.file_path] = ref_rec
    for per_rec in rhy_data_read[exercise]['per']:
        path2object_dict[per_rec.file_path] = per_rec
        
new_data_file_name = '../data/rhythm/rhythm-data_file2object.pickle'
with open(new_data_file_name, 'wb') as handle:
    pickle.dump(path2object_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

