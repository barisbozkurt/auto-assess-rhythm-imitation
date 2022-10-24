# -*- coding: utf-8 -*-
"""group_rhythm_features.py

Reads rhythm-data.pickle file created using datapreprocess_rhythm.py and 
test and train file lists created using convert_annotations.py and creates
containers that could be used in machine learning tests

@author: barisbozkurt
"""

import os
import shutil
import numpy as np
import urllib.request
from textdistance import levenshtein, damerau_levenshtein, jaro, jaro_winkler
from scipy import spatial, stats
from scipy.spatial.distance import yule, matching, hamming
import pickle
import time
from datapreprocess_rhythm import Recording

SAMPLE_RATE = 44100
WINDOW_SIZE = 1024
HOP_SIZE = 512
WINDOWING_METHOD = 'hann'

THRESH_ONSET_RATIO = 1 # threshold with respect to mean of ODF

# used for quantizing purposes
ONSET_N_OF_BINS = 60
ODF_LEN = 128


# DISTANCE FUNCTIONS
def mean_squared_error_for_rhythm(A_bin, E_bin, Km = 25 , Ks = 0.55):
    '''
    Implementation of a MSE function designed
    for rhythmic assessment purposes. The original
    algorithm is proposed by Percival in
    http://percival-music.ca/research/masters-percival.pdf
    
    @author: felipevieira
    '''
    def closest_value(alist, value):
        min_diff = float('inf')
        c_value = None
        
        for element in alist:
            if abs(value - element) < min_diff:
                min_diff = abs(value - element)
                c_value = element
        
        return c_value
                
        
    A = [i for i in range(len(A_bin)) if A_bin[i]]
    E = [i for i in range(len(E_bin)) if E_bin[i]]
    
    mse = 0
    
    for i in range(len(E)):
        c_val = closest_value(A, E[i])
        error = abs(c_val - E[i])
        error = np.power(error,2)/len(E)
        error = min(error, Km)
        mse+=error
    
    for i in range(len(A)):
        c_val = closest_value(E, A[i])
        error = abs(c_val - A[i])
        error = np.power(error,2)/len(A)
        error = min(error, Km)
        mse+=error
    
    return 100 - Ks * mse

def beat_difference(v1, v2):
    '''
    Calculates the beat_difference between two vectors

    Parameters
    ----------
    v1: numpy.array
       First vector (eg. reference onsets)
    v2: numpy.array
      Second vector (eg. performance onsets)
        
    Returns
    -------
    onsets : numpy.array
        Onsets vector for the waveform

    @author: felipevieira
    '''
    return abs(np.sum(v1) - np.sum(v2))


def download_rhythm_data_from_zenodo():
    #Download data from Zenodo : https://zenodo.org/record/2620357#.YYPFB3VfgW0
    file_url = "https://zenodo.org/record/2620357/files/MAST_rhy_m4a.zip?download=1"
    zip_file_name = file_url.split('/')[-1].split('?')[0]
    urllib.request.urlretrieve(file_url, zip_file_name)
    #Unpacking the data zip package
    zip_file_name = file_url.split('/')[-1].split('?')[0]
    dba_folder = 'data'
    shutil.unpack_archive(zip_file_name, dba_folder)
    os.remove(zip_file_name)


#-------------         
# DATA preparation for machine learning tests
def read_test_combinations_in_list(file_path):
    test_combinations = []
    ref_files = []
    per_files = []
    with open(file_path) as file:
        for line in file:
            if len(line) > 2:
                test_combinations.append(line.strip().replace('\t',' '))
                ref_file = line.strip().split()[0]
                per_file = line.strip().split()[-1]
                if ref_file not in ref_files:
                    ref_files.append(ref_file)
                if per_file not in per_files:
                    per_files.append(per_file)
    return test_combinations, ref_files, per_files

def add_column_names_to_csv(file_pointer, feature_names_list):
    file_pointer.write('Ref_file,Per_file,')
    for feature_names in feature_names_list:
        for dist_name in feature_names:
            file_pointer.write('{},'.format(dist_name))
    file_pointer.write('grade\n')
    
def add_line_to_csv(file_pointer, ref_file_path, per_file_path, dists_list, grade):
    file_pointer.write(ref_file_path.replace('.wav','') + ',' + per_file_path.replace('.wav','') + ',')
    for dists in dists_list:
        for dist in dists:
            file_pointer.write('{:.6f},'.format(dist))
    file_pointer.write('{}\n'.format(grade)) 

def prepare_ML_data(data_file_name, train_couples_file, test_couples_file, exclude_ref_list_file, onset_method):
    #reading pickle
    with open(data_file_name, 'rb') as handle:
        rhy_data_read = pickle.load(handle)
        
    # Read file lists for test and train 
    train_combinations, train_ref_files, train_per_files = read_test_combinations_in_list(train_couples_file)
    test_combinations, test_ref_files, test_per_files = read_test_combinations_in_list(test_couples_file)
    
    # Read reference files excluded because they received a grade lower than 4
    if os.path.exists(exclude_ref_list_file):
        ref_files_2_exclude = []
        with open(exclude_ref_list_file) as file:
            for line in file:
                if len(line) > 2:
                    ref_files_2_exclude.append(line.strip().split()[0])
        
        # Sanity check: check if excluded ref files exist in target files
        for train_ref_file in train_ref_files:
            if train_ref_file in ref_files_2_exclude:
                print('!!!Error: reference file to be excluded exists in the train set:', train_ref_file)
        
        for test_ref_file in test_ref_files:
            if test_ref_file in ref_files_2_exclude:
                print('!!!Error: reference file to be excluded exists in the test set:', test_ref_file)
    
    DISTANCE_FUNCTIONS_binary = [beat_difference, mean_squared_error_for_rhythm, #rhythm-based features
                              levenshtein, damerau_levenshtein, jaro, jaro_winkler, # text-based features
                              hamming, yule] # vector-based features
    DISTANCE_FUNCTIONS_binary_names = ['bin_beat_diff', 'bin_msq_4_rhy', #rhythm-based features
                              'bin_lev', 'bin_dam_lev', 'bin_jaro', 'bin_jaro_wink', # text-based features
                              'bin_ham', 'bin_yule'] # vector-based features
    
    correlation = spatial.distance.correlation
    cosine = spatial.distance.cosine
    cityblock = spatial.distance.cityblock
    euclidean = spatial.distance.euclidean
    wasserstein = stats.wasserstein_distance  # earth mover distance
    
    DISTANCE_FUNCTIONS_ODF_onsets = [correlation, cosine, cityblock, euclidean, wasserstein] 
    DISTANCE_FUNCTIONS_ODF_names = ['ODF_corr', 'ODF_cos', 'ODF_cityblock', 'ODF_eucl', 'ODF_wass'] 
    DISTANCE_FUNCTIONS_onsets_names = ['onsets_corr', 'onsets_cos', 'onsets_cityblock', 'onsets_eucl', 'onset_wass'] 
    
    # Creating data files comsumed by the base-line system: .csv files containing 
    # distances as features and the grade at the last column
    test_dataFrame_f = 'testData.csv'
    test_dataFrame_f_wr = open(test_dataFrame_f, 'w')
    #Add column names to data frame
    add_column_names_to_csv(test_dataFrame_f_wr, [DISTANCE_FUNCTIONS_binary_names, DISTANCE_FUNCTIONS_ODF_names, DISTANCE_FUNCTIONS_onsets_names])
    
    train_dataFrame_f = 'trainData.csv'
    train_dataFrame_f_wr = open(train_dataFrame_f, 'w')
    #Add column names to data frame
    add_column_names_to_csv(train_dataFrame_f_wr, [DISTANCE_FUNCTIONS_binary_names, DISTANCE_FUNCTIONS_ODF_names, DISTANCE_FUNCTIONS_onsets_names])
    
    # # TODO: Also create data structures to collect ODF, binary onset vectors
    # # each row contains concatenated version of ref vector and per vector
    num_test_samples = len(test_combinations)
    X_test_ODF = np.zeros((num_test_samples, ODF_LEN * 2))
    X_test_binOnsetVect = np.zeros((num_test_samples, ONSET_N_OF_BINS * 2))
    y_test = np.zeros((num_test_samples, ))
    
    num_train_samples = len(train_combinations)
    X_train_ODF = np.zeros((num_train_samples, ODF_LEN * 2))
    X_train_binOnsetVect = np.zeros((num_train_samples, ONSET_N_OF_BINS * 2))
    y_train = np.zeros((num_train_samples, ))
    
    print('\nDistance features extraction from couples starts, ... most time consuming part')
    print('Number of test and train couples to be processed: ', (num_train_samples + num_test_samples))
    test_ind = 0
    train_ind = 0
    train_comb_files = [] # used for debugging purposes
    test_comb_files = [] # used for debugging purposes
    for exercise in rhy_data_read.keys():
        print('Processing data for exercise', exercise)
        for ref_rec in rhy_data_read[exercise]['ref']:
            for per_rec in rhy_data_read[exercise]['per']:
                if ((per_rec.file_path in test_per_files) and # to speed up process: compute if combination is in test data or train data
                    (ref_rec.file_path in test_ref_files) or 
                    (per_rec.file_path in train_per_files) and 
                    (ref_rec.file_path in train_ref_files)):
                    
                    # Compute distances from binary onset vectors
                    ref_bin_onset_vect = ref_rec.binary_onset_vector[onset_method]
                    per_bin_onset_vect = per_rec.binary_onset_vector[onset_method]
                    
                    # Rhythm based distances
                    dists_binaryOnset = [distance_function(ref_bin_onset_vect, per_bin_onset_vect) for distance_function in DISTANCE_FUNCTIONS_binary[:2]]
                    
                    # Text based distances
                    ref_vect_str = ''.join([str(int(val)) for val in ref_bin_onset_vect])
                    per_vect_str = ''.join([str(int(val)) for val in per_bin_onset_vect])
                    dists_binaryOnset += [distance_function(ref_vect_str, per_vect_str) for distance_function in DISTANCE_FUNCTIONS_binary[2:-2]]
                    
                    # Vector based distances over boolean arrays
                    dists_binaryOnset += [distance_function(ref_bin_onset_vect, per_bin_onset_vect) for distance_function in DISTANCE_FUNCTIONS_binary[-2:]]
                    
                    # Compute distances from cropped ODF functions
                    ref_ODF = ref_rec.ODF_crop_fixLen[onset_method]
                    per_ODF = per_rec.ODF_crop_fixLen[onset_method]
                    if ref_ODF.size != per_ODF.size: # TODO: to be corrected, vector size may come out to be 127 or 128
                        min_len = min(ref_ODF.size, per_ODF.size)
                        ref_ODF = ref_ODF[:min_len]
                        per_ODF = per_ODF[:min_len]
                        
                    dists_ODFs = [distance_function(ref_ODF, per_ODF) for distance_function in DISTANCE_FUNCTIONS_ODF_onsets]
                    
                    # Compute distances from cropped onsets
                    ref_onsets = ref_rec.onsets_crop[onset_method]
                    per_onsets = per_rec.onsets_crop[onset_method]
                    
                    # If sizes do not match, pad zeros
                    if per_onsets.size < ref_onsets.size:
                        per_onsets = np.concatenate((per_onsets, np.zeros((ref_onsets.size-per_onsets.size,))))
                    elif per_onsets.size > ref_onsets.size:
                        ref_onsets = np.concatenate((ref_onsets, np.zeros((per_onsets.size-ref_onsets.size,))))
                    dists_onsets = [distance_function(ref_onsets, per_onsets) for distance_function in DISTANCE_FUNCTIONS_ODF_onsets]
        
                    if (per_rec.file_path in test_per_files) and (ref_rec.file_path in test_ref_files): # put data to test set
                        if (ref_rec.file_path + ' ' + per_rec.file_path) in test_combinations:
                            add_line_to_csv(test_dataFrame_f_wr, ref_rec.file_path, per_rec.file_path, 
                                            [dists_binaryOnset, dists_ODFs, dists_onsets], per_rec.grade)
                            X_test_ODF[test_ind] = np.concatenate((ref_rec.ODF_crop_fixLen[onset_method], per_rec.ODF_crop_fixLen[onset_method]))
                            X_test_binOnsetVect[test_ind] = np.concatenate((ref_rec.binary_onset_vector[onset_method], per_rec.binary_onset_vector[onset_method]))
                            y_test[test_ind] = per_rec.grade
                            test_comb_files.append(ref_rec.file_path + ' ' + per_rec.file_path)
                            test_ind += 1
                    elif (per_rec.file_path in train_per_files) and (ref_rec.file_path in train_ref_files):# put data to train set
                        if (ref_rec.file_path + ' ' + per_rec.file_path) in train_combinations:
                            add_line_to_csv(train_dataFrame_f_wr, ref_rec.file_path, per_rec.file_path, 
                                            [dists_binaryOnset, dists_ODFs, dists_onsets], per_rec.grade)
                            X_train_ODF[train_ind] = np.concatenate((ref_rec.ODF_crop_fixLen[onset_method], per_rec.ODF_crop_fixLen[onset_method]))
                            X_train_binOnsetVect[train_ind] = np.concatenate((ref_rec.binary_onset_vector[onset_method], per_rec.binary_onset_vector[onset_method]))
                            y_train[train_ind] = per_rec.grade
                            train_comb_files.append(ref_rec.file_path + ' ' + per_rec.file_path)
                            train_ind += 1
                        
                    if (test_ind + train_ind) > 0 and (test_ind + train_ind) % 5000 == 0:
                        print('Number of data couples saved:', (test_ind + train_ind))
        
    if train_ind < num_train_samples:
        print('Number of train couples in text file: ',num_train_samples)
        print('Number of train couples collected: ',train_ind)
        X_train_ODF = X_train_ODF[:train_ind]
        X_train_binOnsetVect = X_train_binOnsetVect[:train_ind]
        y_train = y_train[:train_ind]
    
    if test_ind < num_test_samples:
        print('Number of test couples in text file: ',num_test_samples)
        print('Number of test couples collected: ',test_ind)
        X_test_ODF = X_test_ODF[:test_ind]
        X_test_binOnsetVect = X_test_binOnsetVect[:test_ind]
        y_test = y_test[:test_ind]
    
    with open('train_ODF_X.npy', 'wb') as f:
        np.save(f, X_train_ODF)
    with open('train_binOnsetVect_X.npy', 'wb') as f:
        np.save(f, X_train_binOnsetVect)
    with open('train_y.npy', 'wb') as f:
        np.save(f, y_train) 
    
    with open('test_ODF_X.npy', 'wb') as f:
        np.save(f, X_test_ODF)
    with open('test_binOnsetVect_X.npy', 'wb') as f:
        np.save(f, X_test_binOnsetVect)
    with open('test_y.npy', 'wb') as f:
        np.save(f, y_test) 
    
    # Storing file names for debugging purposes
    with open('test_train_file_names.pickle', 'wb') as handle:
        pickle.dump((train_comb_files,test_comb_files), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    test_dataFrame_f_wr.close()
    train_dataFrame_f_wr.close()
    
    # To create a sample feature plot:
    # rhy_data_read['52_rhy2']['ref'][0].plot_features(True)

def main():    
    # Reading and accessing features
    data_file_name = '../data/rhythm/rhythm-data.pickle'
    train_couples_file = '../data/rhythm/train_couples_balanced.txt'
    test_couples_file = '../data/rhythm/test_couples_balanced.txt'
    onset_method = 'hfc'
    exclude_ref_list_file = 'excluded_references.txt' 
    
    prepare_ML_data(data_file_name, train_couples_file, test_couples_file, exclude_ref_list_file, onset_method)


if __name__ == "__main__":
    main()
