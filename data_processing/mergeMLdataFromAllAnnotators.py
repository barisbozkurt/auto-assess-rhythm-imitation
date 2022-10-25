#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merges ML data in csv files by all 3 annotators to get a subset of the data
with high agreement of the annotators. Two modes are defined and used:
    'full_agree': all annotators' labels match exactly
    'majority': majority voting of grades assigned as the grade

Runs the process on trainData.csv and testData.csv files available in zip packs
    ['rhythm_data4ML_0.zip', 'rhythm_data4ML_1.zip', 'rhythm_data4ML_2.zip']
and creates new .zip files at the same folder containing data files named:
    ['rhythm_data4ML_full_agree.zip','rhythm_data4ML_majority.zip']

@author: barisbozkurt
"""

import os, shutil, zipfile
import numpy as np
import pandas as pd

def unpack_files(root_dba_folder, data_pack_files):
    '''Unzips packages in data_pack_files'''
    for zip_file_name in data_pack_files:
        zip_file_name = os.path.join(root_dba_folder, zip_file_name)
        data_folder = zip_file_name.replace('.zip','')
        shutil.unpack_archive(zip_file_name, data_folder)        

def read_all_annots(root_dba_folder, data_pack_files):
    '''Reads all annotations from 3 experts from ''all_annots_2015_2016_rhy.txt'
    in packages and returns a list carrying 3 dictionaries that map 
    file-name to grade'''
    name_2_grade_maps = [] # list to carry dictionaries
    for zip_file_name in data_pack_files:
        data_folder = os.path.join(root_dba_folder, zip_file_name).replace('.zip','')
        name_2_grade = {}
        with open(os.path.join(data_folder, 'all_annots_2015_2016_rhy.txt')) as list_file:
            for line in list_file:
                file = line.split()[0].strip().replace('.wav','')
                grade = int(line.strip().split('Grade:')[-1])
                name_2_grade[file] = grade
        name_2_grade_maps.append(name_2_grade)
    return name_2_grade_maps

def merge_annots(root_dba_folder, data_pack_files, mode='full_agree'):
    '''Given the root folder and data-package zip file names in a list,
    reads all csv files, merges data with full-agreement or majority voting 
    (defined by input variable 'mode'') and stores merged tabular data in 
     a new package'''

    # Read annotations by 3 experts in a list containing dictionaries
    name_2_grade_maps = read_all_annots(root_dba_folder, data_pack_files)
    
    final_map = {}
    for file_name, grade0 in name_2_grade_maps[0].items():

        if file_name in name_2_grade_maps[1] and file_name in name_2_grade_maps[2]:
            grade1 = name_2_grade_maps[1][file_name]
            grade2 = name_2_grade_maps[2][file_name]
            if mode == 'full_agree':
                if grade0 == grade1 and grade0 == grade2:
                    final_map[file_name.replace('.wav','')] = grade0
            elif mode == 'majority':
                grades = [grade0, grade1, grade2]
                counts = np.bincount(grades)
                most_common = np.argmax(counts)
                if counts[most_common] >= 2:
                    final_map[file_name.replace('.wav','')] = most_common
    return final_map

def consistency_check(df_filt, root_dba_folder, data_pack_files, mode='full_agree'):
    '''Checks consistency of df_filt and annotations in data_pack_files
    Written for debugging purposes'''    
    # Read annotations by 3 experts in a list containing dictionaries
    name_2_grade_maps = read_all_annots(root_dba_folder, data_pack_files)
    
    for index, row in df_filt.iterrows():
        df_grade = row['grade']
        annot_grades = np.array([name_2_grade_maps[i][row['Per_file']] for i in range(3)])
        diff_grades = np.abs(annot_grades - df_grade)
        mismatchFound = False
        if mode == 'full_agree':
            if np.sum(diff_grades) != 0:
                mismatchFound = True
        elif mode == 'majority':
            num_matches = np.bincount(diff_grades)[0]
            if num_matches < 2:
                mismatchFound = True
        
        if mismatchFound: # print error if mismatch found
            print('Error for file {}'.format(row['Per_file']))
            print(mode, 'rule does not apply, df_grade:{}, annot_grades:{}'.format(df_grade, annot_grades))
        
#----------------
def main():
    root_dba_folder = '../data/'        
    data_pack_files = ['rhythm_data4ML_0_withCorrectedOnsets.zip', 
                       'rhythm_data4ML_1_withCorrectedOnsets.zip',
                       'rhythm_data4ML_2_withCorrectedOnsets.zip']
    # Unpacks zip files to folders
    unpack_files(root_dba_folder, data_pack_files)
    
    csv_files = ['trainData.csv','testData.csv']
    for mode in ['full_agree', 'majority']:
        # Create a dictionary that carries merged map file_name -> grade
        #   applying the 'full_agree' or 'majority' rule
        name_2_grade_map = merge_annots(root_dba_folder, data_pack_files, mode=mode)
        
        # For train and test data csv files
        for file_name in csv_files:
            # Reading csv file of the first annotator (data_pack_files[0])
            csv_file = root_dba_folder + data_pack_files[0].replace('.zip','')+'/'+file_name
            df = pd.read_csv(csv_file)
            
            # Find lines in the first annotator's data that for which
            #  both ref_file and per_file exists in name_2_grade_map 
            indexes_2_keep = []
            for ind in range(df.shape[0]):
                if (df.iloc[ind]['Ref_file'] in name_2_grade_map and
                    df.iloc[ind]['Per_file'] in name_2_grade_map):
                    indexes_2_keep.append(ind)
            
            # Reduce data frame of the first annotator to lines containing 
            indexes_2_keep = np.array(indexes_2_keep)
            
            # Create a new dataframe containing a subset of rows of the original
            #   and modifying its grade
            df_filt = df.iloc[indexes_2_keep].copy(deep=True)
            # Change grade to that of name_2_grade_map - 'Per_file'
            for ind in df_filt.index:
                df_filt.loc[ind, 'grade'] = int(name_2_grade_map[df_filt.loc[ind]['Per_file']])
            
            # Consistency check (for debugging purposes)
            consistency_check(df_filt, root_dba_folder, data_pack_files, mode=mode)
            
            # write to file
            df_filt.to_csv(file_name, index=False)
            
        # Packing ML data
        ML_data_folder = root_dba_folder + 'rhythm_data4ML_' + mode + '/'
        os.mkdir(ML_data_folder)
        for file in csv_files:
            shutil.copyfile(file, os.path.join(ML_data_folder, file))
            os.remove(file)
        
        zip_file_name = 'rhythm_data4ML_' + mode
        shutil.make_archive(os.path.join(root_dba_folder, zip_file_name), 'zip', ML_data_folder)
        
        # Delete ML data folder
        shutil.rmtree(ML_data_folder)
    
    # Delete ML data folders created    
    for zip_file_name in data_pack_files:
        zip_file_name = os.path.join(root_dba_folder, zip_file_name)
        data_folder = zip_file_name.replace('.zip','')
        shutil.rmtree(data_folder)
    
if __name__ == '__main__':
   main()