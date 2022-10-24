#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to plot sample features
Used for debugging purposes

@author: barisbozkurt
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

def select_indexes(y_data, num_samples_per_grade):
    selected_inds = []
    count_per_grade = np.array([0,0,0,0]) # counters for num samples per grade
    num_samples = y_data.size
    while True:
        ind = random.randint(0, num_samples-1)
        if ind not in selected_inds:
            grade_sample = y_data[ind]
            if count_per_grade[grade_sample-1] < num_samples_per_grade:
                selected_inds.append(ind)
                count_per_grade[grade_sample-1] += 1
        #check if all samples selected
        if np.sum(count_per_grade) == num_samples_per_grade*4:
            break
    return selected_inds

def select_plot_samples(y, x_ODF, x_binOnset, file_names, num_samples_per_grade):
    #Selecting samples
    selected_inds = select_indexes(y, num_samples_per_grade)
    
    for ind in selected_inds:
        plt.figure()
        plt.subplot(2,1,1)
        plt.title('Grade:' + str(y[ind]) + '-' + file_names[ind])
        feature = x_ODF[ind]
        plt.plot(feature[:feature.size//2],'r')
        plt.plot(feature[feature.size//2:],'b')
        plt.subplot(2,1,2)
        feature = x_binOnset[ind]
        plt.stem(feature[:feature.size//2], markerfmt='r.')
        plt.stem(feature[feature.size//2:], markerfmt='bo')
        plt.savefig('figures/' + str(y[ind]) + '-' + file_names[ind].replace(' ','_')+'.png')
        plt.close()
        
    

y_test = np.load('test_y.npy').astype(int)
x_test_ODF = np.load('test_ODF_X.npy')
x_test_binOnset = np.atleast_3d(np.load('test_binOnsetVect_X.npy'))

y_train = np.load('train_y.npy').astype(int)
x_train_ODF = np.load('train_ODF_X.npy')
x_train_binOnset = np.load('train_binOnsetVect_X.npy')

with open('test_train_file_names.pickle', 'rb') as handle:
    train_comb_files,test_comb_files = pickle.load(handle)

num_samples_per_grade = 4
if not os.path.exists('figures'):
    os.mkdir('figures')
select_plot_samples(y_test, x_test_ODF, x_test_binOnset, test_comb_files, num_samples_per_grade)
select_plot_samples(y_train, x_train_ODF, x_train_binOnset, train_comb_files, num_samples_per_grade)

