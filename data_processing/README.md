# Data preprocessing and feature extraction

This folder contains data processing scripts that process audio and annotation 
files to prepare data for machine learning experiments in the form of tabular data 
saved in cvs files or matrices stored in .npy files. 

Before running the individual scripts to process data you should place all rhythm wav 
files (avilable on: https://zenodo.org/record/2620357#.YYPFB3VfgW0) under data/wav/ 
which could be done via running downloadAudioFromZenodo.py or 
dataProcessPipeline.py

dataProcessPipeline.py runs all data preparation steps and runs a seqeunce 
of functions from the following scripts:

*   downloadAudioFromZenodo.py downloads all audio data from Zenodo and converts .m4a files to .wav files. Places all resulting files in data/wav folder
 
*   convert_annotations.py: reads annotation files created by data annotation tool, merges them. Produces train, test split of data and produces file list files.

*   datapreprocess_rhythm.py: Defines a Recording class which includes 
    onset detection functions, the required ODF and onset instance variables, functions to 
    compute them and a main function that produces Recording object for each file, groups them 
    according to the question ID and stores all in a pickle file (rhythm-data.pickle)
    It is set to use the HFC method for automatic onset detection. It stores
    onsets estimated in *.os.txt files at the same folder as the wave files. If you would
    like to use manually corrected onsets; place the *.os.txt files in package 
    manuallyCorrectedOnsets.zip at the same location as the wave files. Then the onset 
    detection would be skipped and the information would be read from the existing files 
    (see code block that starts with: if os.path.exists(self.onset_file) in _extract_onsets())

*   group_rhythm_features.py: Reads rhythm-data.pickle file and test and train 
    file lists created using convert_annotations.py and creates feature-output data used in machine 
    learning tests

*   visualize_rhythm_features.py: Plotting functions to check rhythm data ODFs and onsets. Used for debugging purposes

When finished, dataProcessPipeline.py creates zip packages in the data folder that contain ML data 
for each annotator. 

