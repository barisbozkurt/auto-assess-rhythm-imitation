
# Siamese Network

This folder contains codes for automatic assessment for rhythmic pattern imitations using Siamese network.

The folder contains the following sub-folders:
* 'experiments' folder contains codes to train/test Siamese network for performance assessment and DNN for onset detection. 
* 'models' folder contains DNN topologies we tried for the performance assessment and the onset detection.
* 'dataset' folder contains codes to prepare the files in the database to train the models. 
* 'callback' folder contains codes to evaluate the performance of the network during the training.

In order to run the codes under experiments directory, features that will be used in the model training should be saved to a directory. 
The name of this database directory should be specified in the configuration.py as follows;

* audio_path_trn_per: Database directory for train performance files.
* audio_path_test_per: Database directory for test performance files.
* audio_path_ref_per: : Database directory for reference files.

If onset features are fed into Siamese network, then the features can be obtained using the codes under ../data_processing folder.
Or, onset labels could be fed into the network. Moreover, Siamese network can use the onset points obtained with experiments/train_onset.py. experiments/train_onset.py implements DNN-based
onset detection in the paper. 

We train DNN-based onset detection with spectrogram features. Spectrogram can be obtained using compute_spectrogram method in utils.py.
The input to the DNN-based onset detection is onset labels. However, we pre-process the onset labels using the get_binary_labels method in utils.py.



