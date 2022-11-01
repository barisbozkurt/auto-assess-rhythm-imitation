#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:44:50 2022

@author: basak
"""

from zipfile import ZipFile
import os
from pathlib import Path

import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np


def extract_zip(zipfile_path, target_folder_path):
    zip = ZipFile(zipfile_path, 'r')
    zip.extractall(target_folder_path)
    
zipfile_path = os.path.join(Path(os.getcwd()).parent,'rhythm/2015-2016Etiketlemeler.zip')
temp = zipfile_path.split(".")
target_folder_path = os.path.join(temp[0])
extract_zip(zipfile_path, target_folder_path)

dirname = os.path.join(Path(os.getcwd()).parent,'rhythm/2015-2016Etiketlemeler')
def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders
subfolders = fast_scandir(dirname)


file_list = []

for mypath in subfolders:

    files = ([f for f in listdir(mypath) if isfile(join(mypath, f))])
    for f in files:
        file_list.append(join(mypath,f))


annotations = pd.DataFrame(columns = ['performance_name','ozan','cihan','aslı'])


for file_name in file_list:
    with open(file_name) as file:
        for line in file:
            temp = line.split(':')
            grade = temp[1][0]
            temp = temp[0].split('/')
            temp = temp[2].split('.')
            performance_name = temp[0]
            annotations.loc[annotations.shape[0]] = [performance_name, grade, '','']
            
annotations.set_index('performance_name', inplace=True)

zipfile_path = os.path.join(Path(os.getcwd()).parent,'rhythm/2015-2016Etiketlemeler_Asli.zip')
temp = zipfile_path.split(".")
target_folder_path = os.path.join(temp[0])
extract_zip(zipfile_path, target_folder_path)

dirname = os.path.join(Path(os.getcwd()).parent,'rhythm/2015-2016Etiketlemeler_Asli')
def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders
subfolders = fast_scandir(dirname)

from os import listdir
from os.path import isfile, join
import numpy as np

file_list = []

for mypath in subfolders:

    files = ([f for f in listdir(mypath) if isfile(join(mypath, f))])
    for f in files:
        file_list.append(join(mypath,f))
        

for file_name in file_list:
    with open(file_name) as file:
        for line in file:
            temp = line.split(':')
            grade = temp[1][0]
            temp = temp[0].split('\\')
            temp = temp[2].split('.')
            performance_name = temp[0]
            annotations.at[performance_name,'aslı']=grade
            

zipfile_path = os.path.join(Path(os.getcwd()).parent,'rhythm/2015-2016Etiketlemeler_Cihan.zip')
temp = zipfile_path.split(".")
target_folder_path = os.path.join(temp[0])
extract_zip(zipfile_path, target_folder_path)

dirname = os.path.join(Path(os.getcwd()).parent,'rhythm/2015-2016Etiketlemeler_Cihan')
def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders
subfolders = fast_scandir(dirname)

from os import listdir
from os.path import isfile, join
import numpy as np

file_list = []

for mypath in subfolders:

    files = ([f for f in listdir(mypath) if isfile(join(mypath, f))])
    for f in files:
        file_list.append(join(mypath,f))
        

for file_name in file_list:
    with open(file_name) as file:
        for line in file:
            temp = line.split(':')
            grade = temp[1][0]
            temp = temp[0].split('\\')
            temp = temp[2].split('.')
            performance_name = temp[0]
            annotations.at[performance_name,'cihan']=grade


            
from nltk import agreement


coder1 = annotations['ozan']
coder2 = annotations['aslı']
coder3 = annotations['cihan']
number_of_sample = annotations.shape[0]
formatted_codes = [[1,i,coder1[i]] for i in range(number_of_sample)] + [[2,i,coder2[i]] for i in range(number_of_sample)]  + [[3,i,coder3[i]] for i in range(number_of_sample)] 


ratingtask = agreement.AnnotationTask(data=formatted_codes)

print('Fleiss\'s Kappa:',ratingtask.multi_kappa())

from statsmodels.stats.inter_rater import fleiss_kappa
annotations_array = annotations.to_numpy()

for i in range(0,annotations_array.shape[0]):
    for j in range(0,annotations_array.shape[1]):
        try:
            annotations_array[i,j] = int(annotations_array[i,j]) 
        except ValueError:
            annotations_array_new = np.delete(annotations_array, i, 0)
            
fleiss_kappa(annotations_array_new)

from sklearn.metrics import cohen_kappa_score
labeler1 = annotations_array_new[:,0].astype(np.float)
labeler2 = annotations_array_new[:,1].astype(np.float)
labeler3 = annotations_array_new[:,2].astype(np.float)
print('Cohen Kappa score between Ozan and Aslı: ', cohen_kappa_score(labeler1, labeler2))
print('Cohen Kappa score between Ozan and Cihan: ', cohen_kappa_score(labeler1, labeler3))
print('Cohen Kappa score between Aslı and Cihan: ', cohen_kappa_score(labeler2, labeler3))