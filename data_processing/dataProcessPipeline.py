#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data process pipeline:
    Given audio data in (../data/wav/) and annotations (in ../data/) runs sequence of processes
    to create tabular data for machine learning experiments

Some processes differ for different annotator's data:
    - To have enough data for DNN experiments for annotator-0,
    exclusivity between test and train sets is guarenteed only for student 
    performance recordings. Reference recordings are common in train and test
    - For annotators 1 and 2, the test and train sets are exclusive for both 
    reference and student performance files
    

@author: barisbozkurt
"""

import os, shutil
import zipfile
import time
import pandas as pd
from downloadAudioFromZenodo import download_rhythm_data_from_zenodo
from convert_annotations import produce_file_lists, create_test_train_split_file_lists
from datapreprocess_rhythm import create_recording_objects
from group_rhythm_features import prepare_ML_data

def consistency_check_annots(tabular_data_files, annotations_file):
    '''Checks consistency of annotations and final tabular data
        Written for debugging purposes
    '''
    num_errors = 0
    # Read annoations file into a dict mapping filename-> grade
    annots_dict = {}
    with open(annotations_file) as list_file:
        for line in list_file:
            file = line.split()[0].strip().replace('.wav','')
            grade = int(line.strip().split('Grade:')[-1])
            annots_dict[file] = grade
    
    # Read ML data, check if annotations match
    # Also check if question matches for ref-per files
    for tabular_data_file in tabular_data_files:
        line_ind = 0
        with open(tabular_data_file) as list_file:
            for line in list_file:
                if line_ind != 0: # skip first line
                    line_splits = line.split(',')
                    ref_file = line_splits[0].strip()
                    per_file = line_splits[1].strip()
                    grade = int(line_splits[-1].strip())
                    
                    # Check consistency of question/pattern
                    pattern_ref = '_'.join(ref_file.split('_')[:2])
                    pattern_per = '_'.join(per_file.split('_')[:2])
                    if pattern_ref != pattern_per:
                        num_errors += 1
                        print('!Error, question/pattern do not match')
                        print('Line {} in {}'.format(line_ind, tabular_data_file))
                    
                    # Checking consistency of grades in two files
                    if grade != annots_dict[per_file]:
                        num_errors += 1
                        print('!Error, grades do not match')
                        print('Line {} in {} has grade {}'.format(line_ind, tabular_data_file, grade))
                        print(' where {} for file {} has grade {}'.format(annotations_file, per_file, annots_dict[per_file]))
                
                line_ind += 1
    if num_errors == 0:
        print('Consistency of annotations and final tabular data checked: OK')
    
def exclusivity_check_ML_data(test_data_file, train_data_file, check_ref_files=False):
    '''Checks exclusivity of test and train sets'''
    num_errors = 0
    test_ref_files = []
    test_per_files = []
    line_ind = 0
    with open(test_data_file) as list_file:
        for line in list_file:
            if line_ind != 0: # skip first line
                line_splits = line.split(',')
                test_ref_files.append(line_splits[0].strip())
                test_per_files.append(line_splits[1].strip())
            line_ind += 1
    
    line_ind = 0
    with open(train_data_file) as list_file:
        for line in list_file:
            if line_ind != 0: # skip first line
                line_splits = line.split(',')
                train_ref_file = line_splits[0].strip()
                train_per_file = line_splits[1].strip()
                # Checking exclusivity of student performance files
                if train_per_file in test_per_files:
                    num_errors += 1
                    print('Error:')
                    print(' Train per file exists also in test set:', train_per_file)
                # Checking exclusivity of teacher/reference files
                if check_ref_files:
                    if train_ref_file in test_ref_files:
                        num_errors += 1
                        print('Error:')
                        print(' Train ref file exists also in test set:', train_ref_file)

            line_ind += 1
            
    if num_errors == 0:
        print('Exclusivity of train and test data checked: OK')
    
def delete_all_files_in_folder(audio_data_folder, search_str):
    '''Deletes all files that contains the search_str in the file name''' 
    for root, dirs, files in os.walk(audio_data_folder):
        for file in files:
            if search_str in file: 
                os.remove(os.path.join(root, file))
                
#---------------------------
start_time = time.time()
annot_data_folder = '../data/'
audio_data_folder = '../data/wav/'

# Check availability of audio data, if not exists: download and convert files to .wav
if not os.path.exists(audio_data_folder):
    os.mkdir(audio_data_folder)
    download_rhythm_data_from_zenodo(audio_data_folder.replace('wav/',''))

annotation_file_packages = ['annotations_0.zip',
                    'annotations_1.zip',
                    'annotations_2.zip']

corrected_onsets_file = os.path.join(annot_data_folder, 'manuallyCorrectedOnsets.zip')

file_list_files = ['test_couples.txt','train_couples.txt']

files_to_pack_in_ML_data = ['all_annots_2015_2016_rhy.txt',
                            'excluded_references.txt',
                            'testData.csv',
                            'trainData.csv',
                            'rhythm-data.pickle']

script_folder = os.path.abspath(os.getcwd())
for annotator_ind, annotation_file_package in enumerate(annotation_file_packages):
    print('------------------------')
    print('PREPARING DATA OF ANNOTATOR ', annotator_ind, ':', annotation_file_package)
    unzip_folder = annot_data_folder + 'temp/'
    with zipfile.ZipFile(os.path.join(annot_data_folder, annotation_file_package),
                         'r') as zip_ref:
        zip_ref.extractall(unzip_folder)

    # Merge annotation files and create balanced test-train splits    
    produce_file_lists(unzip_folder)
    create_test_train_split_file_lists(script_folder)
    # Delete the temporary folder created for annotation files
    shutil.rmtree(unzip_folder)
    
    if annotator_ind == 0:
        # If test-train file lists already exist in the main data folder, 
        #   use them instead, i.e. overwrite those files
        for source_file in file_list_files:
            source_file_path = os.path.join(annot_data_folder, source_file)
            if os.path.exists(source_file_path):
                target_file_path = os.path.join(script_folder, source_file)
                shutil.copyfile(source_file_path, target_file_path)
    
    for mode in ['estimateOnsets','useCorrectedOnsets']:
        print('MODE:', mode)
        
        # Delete all os.txt files in audio folder if onsets will be estimated from scratch
        # else copy corrected onset files to audio folder 
        if mode == 'estimateOnsets':
            search_str = '.os.txt'
            delete_all_files_in_folder(audio_data_folder, search_str)
            print('Re-estimating onsets from scratch')
        else:
            with zipfile.ZipFile(corrected_onsets_file,'r') as zip_ref:
                zip_ref.extractall(audio_data_folder)
            print('Process uses manually corrected onsets ')

        
        target_rec_objects_file = os.path.join(script_folder, 'rhythm-data.pickle')
        annotations_file = 'all_annots_2015_2016_rhy.txt'
        
        # Creating recording objects: runs onset detection
        create_recording_objects(target_rec_objects_file, audio_data_folder, annotations_file)
        
        # Grouping data and preparing data files ready for ML experiments
        train_couples_file = os.path.join(script_folder,'train_couples.txt')
        test_couples_file = os.path.join(script_folder,'test_couples.txt')
        onset_method = 'hfc'
        exclude_ref_list_file = os.path.join(script_folder,'excluded_references.txt')
        
        prepare_ML_data(target_rec_objects_file, train_couples_file, test_couples_file, exclude_ref_list_file, onset_method)
        
        # Checks consistency of annotations and final tabular data
        consistency_check_annots(['testData.csv','trainData.csv'], annotations_file)
        
        # Check exclusivity of train and test data
        if annotator_ind == 0:
            exclusivity_check_ML_data('testData.csv','trainData.csv')
        else:
            exclusivity_check_ML_data('testData.csv','trainData.csv', check_ref_files=True)
        
        # Packing ML data
        ML_data_folder = annot_data_folder + 'rhythm_data4ML/'
        os.mkdir(ML_data_folder)
        for file in files_to_pack_in_ML_data:
            shutil.copyfile(file, os.path.join(ML_data_folder, file))
        
        zip_file_name = 'rhythm_data4ML_'+str(annotator_ind)
        if mode == 'useCorrectedOnsets':
            zip_file_name += '_withCorrectedOnsets'
        shutil.make_archive(os.path.join(annot_data_folder, zip_file_name), 'zip', ML_data_folder)
        
        # Delete ML data folder
        shutil.rmtree(ML_data_folder)
    

stop_time = time.time()
print('Total duration for data preparation: ', (stop_time - start_time)/60, 'minutes')
