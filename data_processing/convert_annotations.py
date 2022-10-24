#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads rhythm annotation files, converts and saves them in new file-list files

Performs split for test and train and produces list files for those

@author: barisbozkurt
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import random

def rename_files(dba_folder, search_str, replace_str):
    '''Renames all files in a given directory
    Used for renaming manually corrected onset files: .txt -> .os.txt
    '''
    for root, dirs, files in os.walk(dba_folder):
        for file in files:
            if search_str in file: # annotation files starts with 'report'
                file_path = os.path.join(root, file)
                target_path = file_path.replace(search_str, replace_str)
                os.rename(file_path, target_path)
                
def produce_file_lists(dba_folder):
    '''
    Given the annotation files produced by the annotation software in subfolders
    of dba_folder, produces list files

    Parameters
    ----------
    dba_folder : str
        Path to main folder containing annotation files.

    Returns
    -------
    None
    Produces new text files containing list of files

    '''
    out_file = 'all_annots_2015_2016_rhy.txt'
    out_file_wr = open(out_file, 'w')

    exc_file = 'excluded_references.txt'
    exc_file_wr = open(exc_file, 'w')
    
    list_file_ref = 'listreferences.txt'
    list_file_ref_wr = open(list_file_ref, 'w')

    list_file_per = 'listperformances.txt'
    list_file_per_wr = open(list_file_per, 'w')
    
    grd_4_per_files = 'listperformances_grd4.txt'
    grd_3_per_files = 'listperformances_grd3.txt'
    grd_2_per_files = 'listperformances_grd2.txt'
    grd_1_per_files = 'listperformances_grd1.txt'
    grd_4_per_files_wr = open(grd_4_per_files, 'w')
    grd_3_per_files_wr = open(grd_3_per_files, 'w')
    grd_2_per_files_wr = open(grd_2_per_files, 'w')
    grd_1_per_files_wr = open(grd_1_per_files, 'w')
    
    grd_4_per_files_list = []
    grd_3_per_files_list = []
    grd_2_per_files_list = []
    grd_1_per_files_list = []
    
    num_performances = 0
    
    annot_dict = {}
    for root, dirs, files in os.walk(dba_folder):
        for file in files:
            if 'report' in file: # annotation files starts with 'report'
                file_path = os.path.join(root, file)
                with open(file_path) as f:
                    for line in f:
                      audio_file = line.strip().split()[0].split('/')[-1]
                      audio_file = audio_file.split("\\")[-1] # removing folder info from filename(windows)
                      grade = int(line.strip().split()[-1].split('Grade:')[-1])
                      annot_dict[audio_file] = grade
                      if 'ref' in audio_file and grade < 4:
                          exc_file_wr.write(audio_file + '\tGrade:' + str(grade) + '\n')                          
                      else:
                          out_file_wr.write(audio_file + '\tGrade:' + str(grade) + '\n')
                          if 'per' in audio_file:
                              num_performances += 1
                              list_file_per_wr.write(audio_file + '\n')
                              if grade == 4:
                                  grd_4_per_files_wr.write(audio_file + '\n')
                                  grd_4_per_files_list.append((audio_file, grade))
                              elif grade == 3:
                                  grd_3_per_files_wr.write(audio_file + '\n')
                                  grd_3_per_files_list.append((audio_file, grade))
                              elif grade == 2:
                                  grd_2_per_files_wr.write(audio_file + '\n')
                                  grd_2_per_files_list.append((audio_file, grade))
                              elif grade == 1:
                                  grd_1_per_files_wr.write(audio_file + '\n')
                                  grd_1_per_files_list.append((audio_file, grade))
                          if 'ref' in audio_file:
                              list_file_ref_wr.write(audio_file + '\n')
                  
    print('Number of student performances:', num_performances)
    out_file_wr.close()
    list_file_ref_wr.close()
    list_file_per_wr.close()
    exc_file_wr.close()
    
    grd_4_per_files_wr.close()
    grd_3_per_files_wr.close()
    grd_2_per_files_wr.close()
    grd_1_per_files_wr.close()
    
    

def write_list_to_file(list_2_write, file_path):
    f_wr = open(file_path, 'w')
    for element in list_2_write:
        f_wr.write(element + '\n')
    f_wr.close()
    
def create_test_train_split_file_lists(list_files_folder, test_split_ratio=0.3):
    '''
    Given file list files created using the produce_file_lists() function,
    performs a split of ref and per recordings making sure the resulting 
    coupled data is balanced in terms of grades. Outputs are written into new 
    text files. The tricky part of the code is creating couples using ref and 
    per samples and producing a balanced and mutually exclusive set both for ref
    and for per samples
    
    test_split_ratio applied to minimum number of files per grade, hence does not
    represent a split ratio on the whole set
    
    '''
    # Create output folder if not exists
    if not os.path.exists(list_files_folder):
        os.mkdir(list_files_folder)
        
    num_files_per_grade = {}
    file_list_per_grade = {}
    for grade in [1,2,3,4]:
        list_file = os.path.join(list_files_folder, 'listperformances_grd{}.txt'.format(grade))
        file_list = []
        with open(list_file) as file_reader:
            for line in file_reader:
                if len(line) > 2:
                    file_list.append(line.strip())
        num_files_per_grade[grade] = len(file_list)
        file_list_per_grade[grade] = file_list
        
    #Find minimum number of grade occurence
    min_num_files = min(num_files_per_grade.values())
    
    #Define number of files to keep for test set for each grade
    # the rest will be kept for train (later balanced before ML tests)
    num_test_files_per_grade = int(min_num_files * test_split_ratio)
    
    #Pick test files (performances)
    test_files_per_grade = {}
    test_files = []
    for grade, file_list in file_list_per_grade.items():
        random.shuffle(file_list)
        selected_list = []
        num_selected = 0
        
        # Picking samples from different questions
        while num_selected < num_test_files_per_grade:
            selected_question_ids = []
            for file in file_list:
                question_id = '_'.join(file.split('_')[:2])
                if question_id not in selected_question_ids:
                    selected_question_ids.append(question_id)
                    num_selected += 1
                    selected_list.append(file)
                if num_selected >= num_test_files_per_grade:
                    break
        
        test_files_per_grade[grade] = selected_list
        test_files += selected_list
    
    #Pick train files (performances)
    train_files_per_grade = {}
    train_files = []
    for grade, file_list in file_list_per_grade.items():
        random.shuffle(file_list)
        selected_list = []
        num_selected = 0
        selected_question_ids = []
        for file in file_list:
            if file in test_files: # skip file if already in test list
                continue
            else:
                num_selected += 1
                selected_list.append(file)

        train_files_per_grade[grade] = selected_list
        train_files += selected_list    
    
    # Check if any test file appears in train file list
    for file in test_files:
        if file in train_files:
            print('Error in splitting, file exists in both train and test:', file)
            
    # Choose reference files to couple with performance files

    #Read references file list
    ref_file_list = []
    with open(os.path.join(list_files_folder,'listreferences.txt')) as file_reader:
        for line in file_reader:
            if len(line) > 2:
                ref_file_list.append(line.strip())
    
    # Pick test ref files and form test ref-per couples
    test_ref_files = []
    num_ref_for_each_test = 3 # number of reference files for each test file
    test_couples_per_grade = {1:[], 2:[], 3:[], 4:[]}
    for grade, file_list in test_files_per_grade.items():
        for file in file_list:
            num_selected = 0
            question_id = '_'.join(file.split('_')[:2])
            for ref_file in ref_file_list:
                question_id_ref = '_'.join(ref_file.split('_')[:2])
                if question_id == question_id_ref and ref_file not in test_ref_files:
                    test_ref_files.append(ref_file)
                    test_couples_per_grade[grade].append(ref_file+'\t'+file)
                    num_selected += 1
                if num_selected >= num_ref_for_each_test:
                    break
            
    # Train ref files will be all except test ref files
    train_ref_files = []
    for ref_file in ref_file_list:
        if ref_file not in test_ref_files:
            train_ref_files.append(ref_file)
            
    # Create ref-per couples for train
    train_couples_per_grade = {1:[], 2:[], 3:[], 4:[]}
    for grade, file_list in train_files_per_grade.items():
        for file in file_list:
            question_id = '_'.join(file.split('_')[:2])
            for ref_file in train_ref_files:
                question_id_ref = '_'.join(ref_file.split('_')[:2])
                if question_id == question_id_ref:
                    train_couples_per_grade[grade].append(ref_file+'\t'+file)
    
    # Check if any test-ref file appears in train-ref file list
    for file in test_ref_files:
        if file in train_ref_files:
            print('Error in splitting, file exists in both train and test:', file)
    
    # Writing lists to text files
    write_list_to_file(test_files, 'test_per_files.txt')
    write_list_to_file(train_files, 'train_per_files.txt')
    write_list_to_file(test_ref_files, 'test_ref_files.txt')
    write_list_to_file(train_ref_files, 'train_ref_files.txt')    
    # Collapse dictionaries to list to write to file
    test_couples_list = []
    for grade, couple_list in test_couples_per_grade.items():
        test_couples_list += couple_list
    train_couples_list = []
    for grade, couple_list in train_couples_per_grade.items():
        train_couples_list += couple_list
    
    write_list_to_file(test_couples_list, 'test_couples.txt')
    write_list_to_file(train_couples_list, 'train_couples.txt')
    

def main():
    # Before running, first unzip /data/rhythm/2015-2016Etiketlemeler.zip 
    # to a folder which should produce the folder below
    dba_folder = '../data/rhythm/2015-2016Etiketlemeler_Asli/'
    # Produce file lists
    produce_file_lists(dba_folder)
    
    # Perform test-train split and produce new files
    # New files are created at the same location as this script to avoid 
    # overwriting files in the data folder and providing chance to check them 
    create_test_train_split_file_lists(os.path.abspath(os.getcwd()))
    
if __name__ == "__main__":
    main()
