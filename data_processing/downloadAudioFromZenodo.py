#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to download audio data from 
https://zenodo.org/record/2620357#.YYPFB3VfgW0
and convert to .wav

@author: barisbozkurt
"""
import os
import shutil
import urllib.request
import zipfile
from pydub import AudioSegment

def download_rhythm_data_from_zenodo(target_folder, convertM4aToWav=True):
    '''
    Downloads rhythm data from Zenodo. Use it if you don't have the rhythm data

    Returns
    -------
    None. Creates a "data" folder and puts all audio there

    '''
    #Download data from Zenodo : https://zenodo.org/record/2620357#.YYPFB3VfgW0
    file_url = "https://zenodo.org/record/2620357/files/MAST_rhy_m4a.zip?download=1"
    zip_file_name = file_url.split('/')[-1].split('?')[0]
    print('Downloading 281.8Mb zip file fom Zenodo to folder of this script')
    urllib.request.urlretrieve(file_url, zip_file_name)
    #Unpacking the data zip package
    zip_file_name = file_url.split('/')[-1].split('?')[0]
    shutil.unpack_archive(zip_file_name, target_folder)
    os.remove(zip_file_name)
    # Zip file contains a subfolder m4a 
    m4a_folder = target_folder + 'm4a/'
    print('Zip file unpacked to {} and removed'.format(m4a_folder))
    # Audio files will be converted to wav and placed in wav_folder
    wav_folder = target_folder + 'wav/'
    if not os.path.exists(wav_folder):
        os.mkdir(wav_folder)    

    # Deleting file 69_rhy2_per1135742_fail.wav, since it contains a single stroke
    # causes a crash while cropping ODFs from first onset to the last onset
    os.remove(os.path.join(m4a_folder, '69_rhy2_per1135742_fail.m4a'))
    
    # Inform user for the long data processing step
    print('Converting all m4a files to .wav')
    print('To check the process see the data/wav folder( where files are being added to)') 
    print(' and the data/m4a folder (where files are deleted from after conversion and moving)')
    print(' The process will end when all files in data/m4a are converted and moved')
    
    if convertM4aToWav:
        for root, dirs, files in os.walk(m4a_folder):
            for filename in files:
                if '.m4a' in filename: # annotation files starts with 'report'
                    m4a_filename = os.path.join(m4a_folder, filename)
                    wav_filename = os.path.join(wav_folder, filename.replace('.m4a', '.wav'))
                    # Convert to .wav and delete m4a file
                    track = AudioSegment.from_file(m4a_filename,  format= 'm4a')
                    track.export(wav_filename, format='wav')
                    os.remove(m4a_filename)
        # All files converted, delete the m4a folder
        os.removedirs(m4a_folder)

def main():
    target_folder = '../data/'
    download_rhythm_data_from_zenodo(target_folder)

if __name__ == "__main__":
    main()