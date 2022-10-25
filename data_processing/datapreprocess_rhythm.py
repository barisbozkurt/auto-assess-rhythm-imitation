# -*- coding: utf-8 -*-
"""datapreprocess_rhythm.py

Defines a Recording class which includes features (onsets, ODFs) as instance 
variables and creates Recording objects for each audio file, groups files
in terms of exercise id (ex: 51_rhy1) and saves all in a pickle file.

The pickle file produced could be read as follows:
with open(data_file_name, 'rb') as handle:
    rhy_data_read = pickle.load(handle)

rhy_data_read is a dictionary mapping exercise-id to list of ref and per 
Recording objects. 
    exerciseID -> ref-per Recording object lists, kept in a dictionary mapping
    'ref' or 'per' -> Recording object list

Example:
    rhy_data['51_rhy1']['ref'][0]: Recording object of the first '51_rhy1' ref sample

See group_rhythm_features.py for an example of reading data
and accessing its contents

@author: barisbozkurt
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('agg') # required to avoid memory leak creating figures in a loop
import matplotlib.pyplot as plt
import urllib.request
from essentia.standard import MonoLoader, OnsetDetection, Windowing, FFT, CartesianToPolar, Onsets, FrameGenerator, Spectrum, Energy
from essentia import Pool, array
from textdistance import levenshtein, damerau_levenshtein, jaro, jaro_winkler
from scipy import spatial, stats
from scipy.spatial.distance import yule, matching, hamming
import librosa
import resampy
import pickle
import time
import random
import gc

SAMPLE_RATE = 44100
WINDOW_SIZE = 1024
HOP_SIZE = 512
WINDOWING_METHOD = 'hann'

THRESH_ONSET_RATIO = 1 # threshold with respect to mean of ODF

# used for quantizing purposes
ONSET_N_OF_BINS = 60
ODF_LEN = 128

class Recording(object):

    def __init__(self, file_path, grade, create_figures=False):
        
        self.file_path = file_path.split('/')[-1]
        self.onset_file = file_path.replace('.wav','.os.txt')
        self.questionID = '_'.join(self.file_path.split('_')[:2])
        self.grade = grade
        self.is_student_performance = '_per' in self.file_path
        if self.is_student_performance:
            self.performanceID = self.file_path.split('_per')[-1].split('.')[0]
        else:
            self.performanceID = self.file_path.split('_ref')[-1].split('.')[0]
        
        self.pass_fail = int('fail' not in self.file_path)
        
        
        #Read audio file
        audio_sig = MonoLoader(filename = file_path, sampleRate = SAMPLE_RATE)()
        #Amplitude normalization
        audio_sig = audio_sig / np.max(np.abs(audio_sig))
        self._extract_onsets(audio_sig)
        
        # Disabled due to memory concerns
        # self.mels = melspectrogram(audio_sig)
        
        # Creating figures to allow checking onsets for deciding files for manual onset correction
        if create_figures: 
            self.plot_features('/'.join(file_path.split('/')[:-1]), audio_sig)

        
    def _extract_onsets(self, audio_sig):
        '''
        Extract the onset vectors from the waveform
    
        Parameters
        ----------
        waveform: numpy.array
           The waveform of the audio record
    
        Returns
        -------
        onsets : numpy.array
            Onsets vector for the waveform
    
        Slightly modified version of: https://github.com/MTG/essentia/blob/master/src/examples/tutorial/example_onsetdetection.py
        '''
        #Computing onset detection functions
        self.ODF = {} # Onset detection function; method-> array
        self.ODF_crop = {} # Onset detection function cropped from first onset to last; method-> array
        self.ODF_crop_fixLen = {} # Onset detection function cropped and resampled to ODF_LEN
        self.onsets = {} # Onsets; method -> array
        self.onsets_crop = {} # Onsets cropped from first to last and scaled in range [0,1]; method -> array
        self.binary_onset_vector = {} # Binary onset vector
        
        w = Windowing(type=WINDOWING_METHOD)
        fft = FFT()
        c2p = CartesianToPolar()
        onsets = Onsets()
        
        # for method in ['hfc','flux','complex']: # Disabled due to memory concerns
        for method in ['hfc']:
            od = OnsetDetection(method=method)
            pool = Pool()
            for frame in FrameGenerator(audio_sig, frameSize=WINDOW_SIZE):
                mag, phase, = c2p(fft(w(frame)))
                pool.add('features', od(mag, phase))
    
            self.ODF[method] = pool['features']
            if os.path.exists(self.onset_file):
                # Read onsets from existing .txt file
                onsets = []
                with open(self.onset_file) as file_reader:
                    for line in file_reader:
                        if len(line) > 3:
                            onsets.append(float(line.split()[0]))
                self.onsets[method] = np.array(onsets)
                
            else:
                self.onsets[method] = onsets(array([pool['features']]),[1])
                # removing onsets with low energy
                self._filter_onsets(method)
                # Write onsets to text file readable by Audacity for manual correction
                out_file_wr = open(self.onset_file, 'w')
                for onset in self.onsets[method]:
                    out_file_wr.write('{:.2f}\t{:.2f}\to\n'.format(onset, onset))
                out_file_wr.close()            
            
            # creating cropped versions of ODFs and onsets
            self._crop_ODFs_onsets(method)
            # obtain fixed length ODFs from cropped ODFs
            self._post_filter_cropped_ODF(method)
            # creating binary onset vector
            ODF_binary = np.zeros((ONSET_N_OF_BINS,))
            onset_inds_ODF = np.floor(self.onsets_crop[method]*ONSET_N_OF_BINS).astype(int)
            ODF_binary[onset_inds_ODF[:-1]] = 1
            self.binary_onset_vector[method] = ODF_binary
            


    def _filter_onsets(self, method):
        '''Filtering out onsets with small ODF values'''
        onset_indexes_in_ODF = self.onsets[method] * SAMPLE_RATE / HOP_SIZE
        ODF_threshold = np.mean(self.ODF[method]) * THRESH_ONSET_RATIO
        
        onsets_to_keep = []
        for i in range(onset_indexes_in_ODF.size):
            # if mean of new two samples of ODF is lower than the threshold, remove onset
            ind = int(onset_indexes_in_ODF[i])
            if ind > 2 and ind < (self.ODF[method].size-3) and np.mean(self.ODF[method][ind+1:ind+3]) > ODF_threshold:
                onsets_to_keep.append(True)
            else:
                onsets_to_keep.append(False)
        
        self.onsets[method] = self.onsets[method][onsets_to_keep]

    def _crop_ODFs_onsets(self, method):
        onset_indexes_in_ODF = self.onsets[method] * SAMPLE_RATE / HOP_SIZE
        first_onset_ind = int(onset_indexes_in_ODF[0])
        last_onset_ind = int(onset_indexes_in_ODF[-1])
        self.ODF_crop[method] = self.ODF[method][first_onset_ind:last_onset_ind]
        self.onsets_crop[method] = self.onsets[method] - self.onsets[method][0]
        # normalize cropped onsets
        self.onsets_crop[method] /= self.onsets_crop[method][-1]
        
    def _post_filter_cropped_ODF(self, method):
        if self.ODF_crop[method].size > 0:
            self.ODF_crop_fixLen[method] = resampy.resample(self.ODF_crop[method],self.ODF_crop[method].size, ODF_LEN, axis=-1)
            # amplitude normalize
            self.ODF_crop_fixLen[method] = self.ODF_crop_fixLen[method] / np.max(self.ODF_crop_fixLen[method])
            # Resamp may return one element less, padd zero in that case
            if self.ODF_crop_fixLen[method].size < ODF_LEN:
                self.ODF_crop_fixLen[method] = np.concatenate((self.ODF_crop_fixLen[method], np.zeros((ODF_LEN-self.ODF_crop_fixLen[method].size,))))
        else:
            print('Zero sized ODF_crop, file:', self.file_path)
            self.ODF_crop_fixLen[method] = np.zeros((1,))
        
    def plot_features(self, wav_folder, audio_sig, plotMode=1):
    
        figures_folder = os.path.join(wav_folder,'figures')
        if not os.path.exists(figures_folder):
            os.mkdir(figures_folder)
        plt.figure(figsize=(12,12))
        
        if plotMode==2: # disable previous design of figure
            plt.subplot(4,1,1)
            plt.ylabel('ODFS')
            colors = 'rbkg'; plt_ind = 0; amps = [1, 0.75, 0.5]
            for method, odf in self.ODF.items():
                odf = odf / np.max(odf)
                # plt.plot(odf, colors[plt_ind], label=method)
                plt.plot(odf)
                plt_ind += 1
            
            plt_ind = 0;
            for method, onsets in self.onsets.items():
                onset_indexes = onsets * SAMPLE_RATE / HOP_SIZE
                plt.vlines(onset_indexes, -amps[plt_ind], 0, colors[plt_ind], linewidth=4, label=method+'Onsets')
                plt_ind += 1
            plt.legend()
            plt.xlim((0, odf.size))
    
            plt.subplot(4,1,2)
            plt.ylabel('Cropped ODFS')
            colors = 'rbkg'; plt_ind = 0; 
            for method, odf in self.ODF_crop.items():
                odf = odf / np.max(odf)
                # plt.plot(odf, colors[plt_ind], label=method)
                plt.plot(odf)
                plt_ind += 1
            
            plt_ind = 0;
            for method, onsets in self.onsets_crop.items():
                onset_indexes = np.floor(onsets * odf.size).astype(int)
                onset_indexes = onset_indexes[:-1] # exclude last
                plt.vlines(onset_indexes, -amps[plt_ind], 0, colors[plt_ind], linewidth=4, label=method+'Onsets')
                plt_ind += 1
            plt.legend()
            plt.xlim((0, odf.size))        
     
            
            plt.subplot(4,1,3)
            plt.ylabel('FixLenODF')
            colors = 'rbkg'; plt_ind = 0; 
            for method, odf in self.ODF_crop_fixLen.items():
                odf = odf / np.max(odf)
                plt.plot(odf, colors[plt_ind], label='Cropped-'+method)
                plt_ind += 1
            plt.legend()
            plt.xlim((0, odf.size))        
    
            plt.subplot(4,1,4)
            plt.ylabel('Binary Onset Vector')
            colors = 'rbkg'; plt_ind = 0; 
            for method, odf in self.binary_onset_vector.items():
                plt.stem(odf)
                plt_ind += 1
            plt.xlim((0, odf.size))      
    
            # plt.subplot(4,1,1)
            # plt.imshow(self.mels, aspect='auto')
            # plt.ylabel('MELS')
        elif plotMode==1:
            plt.subplot(2,1,1)
            plt.ylabel('ODFS')
            colors = 'rbkg'; plt_ind = 0; amps = [1, 0.75, 0.5]
            for method, odf in self.ODF.items():
                odf = odf / np.max(odf)
                # plt.plot(odf, colors[plt_ind], label=method)
                plt.plot(odf)
                plt_ind += 1
            
            plt_ind = 0;
            for method, onsets in self.onsets.items():
                onset_indexes = onsets * SAMPLE_RATE / HOP_SIZE
                plt.vlines(onset_indexes, -amps[plt_ind], 0, colors[plt_ind], linewidth=4, label=method+'Onsets')
                plt_ind += 1
            plt.legend()
            plt.xlim((0, odf.size))
            plt.subplot(2,1,2)
            plt.plot(audio_sig)
            method = 'hfc'
            for onsets in self.onsets[method]:
                onset_indexes = onsets * SAMPLE_RATE
                plt.vlines(onset_indexes, -1, 1, 'r', linewidth=4)
                plt_ind += 1
            plt.xlim((0, audio_sig.shape[0]))
            
 
        # plt.savefig(self.file_path.replace('.wav','.eps'), format='eps', dpi=1200)
        plt.savefig(os.path.join(figures_folder, self.file_path.replace('.wav','.png')))
        plt.clf()
        plt.close()
        gc.collect()



def melspectrogram(audio_sig, sr=16000, n_mels=48, n_frames=128, rms_threshold_ratio=0.05, rms_crop_on=False):
  #Crop from two sides using rms threshold
  if rms_crop_on:
    #Computing rmse
    frame_length_rms = int(4 * (audio_sig.size / n_frames)) # 4 times num_frames required for the feature
    hop_length_rms = frame_length_rms // 2
    rms = librosa.feature.rms(y=audio_sig, frame_length=frame_length_rms, hop_length=hop_length_rms)
    rms = rms.ravel()
    rms_threshold = rms.max() * rms_threshold_ratio

    #Strip/crop the signal and the features on two ends based on rms thresholding
    #Finding indexes on left with rms lower than threshold
    ind = 0
    while rms[ind] < rms_threshold:
      ind += 1
    #Cropping vector from left
    audio_sig = audio_sig[ind*hop_length_rms:]
    #Finding indexes on right with rms lower than threshold
    ind = -1
    while rms[ind] < rms_threshold:
      ind -= 1
    #Cropping vector from right
    audio_sig = audio_sig[:ind*hop_length_rms]

  #Compute melspectrogram
  frame_length = int(audio_sig.size/(n_frames+1)) * 2 + 1
  S = librosa.feature.melspectrogram(y=audio_sig, sr=sr, n_fft=frame_length, hop_length=frame_length//2, n_mels=n_mels, fmax=sr/2)
  return librosa.power_to_db(S, ref=np.max)[:,:n_frames]


def create_recording_objects(target_rec_objects_file, audio_files_folder, annotations_file):
    
    num_samples = 0
    # Data structure to be saved: 
    #    exerciseID -> ref-per Recording object lists
    rhy_data = dict()
    start_time = time.time()
    print('Audio feature extraction starts, ... will take a few minutes to finish')
    audio_not_found_counter = 0
    with open(annotations_file) as list_file:
        for line in list_file:
            file = line.split()[0].strip()
            pat_name = '_'.join(file.split('_')[:2])
            grade = int(line.strip().split('Grade:')[-1])
            if pat_name not in rhy_data:
                rhy_data[pat_name] = {}
                rhy_data[pat_name]['ref'] = []
                rhy_data[pat_name]['per'] = []
            
            audio_file_path = os.path.join(audio_files_folder, file)
            if os.path.exists(audio_file_path):
                # Main feature extraction takes place in the next line where Recording object is created
                rec_object = Recording(audio_file_path, grade)
                if rec_object.is_student_performance:
                    rhy_data[pat_name]['per'].append(rec_object)
                else:
                    rhy_data[pat_name]['ref'].append(rec_object)
                
                num_samples += 1
                if num_samples % 500 == 0:
                    print('Num files processed:', num_samples)
            else:
                print('Audio file not available:', audio_file_path)
                audio_not_found_counter += 1
                if audio_not_found_counter > 3:
                    print('!!Audio files could not be found')
                    print('Consider running downloadAudioFromZenodo.py prior to this script')
                    return False
    stop_time = time.time()
    print('Duration for feature extraction: ', (stop_time - start_time)/60, 'minutes')
    print('Total number of recordings analyzed:', num_samples)
    
    with open(target_rec_objects_file, 'wb') as handle:
            pickle.dump(rhy_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print('Data saved to file:', target_rec_objects_file)
    return True

def main():
    # Data file to be created 
    target_rec_objects_file = '../data/rhythm-data.pickle'
    audio_files_folder = '../data/wav/'
    annotations_file = '../data/all_annots_2015_2016_rhy.txt'
    
    succeed = create_recording_objects(target_rec_objects_file, audio_files_folder, annotations_file);
    
    if not succeed:
        print('Could not create Recording objects')


if __name__ == "__main__":
    main()
