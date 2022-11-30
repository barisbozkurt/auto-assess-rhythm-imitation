
# Siamese Network

This folder contains codes for automatic assessment for rhythmic pattern imitations using Siamese network.
It also includes the codes for DNN-based onset detection in the paper.

The folder contains the following sub-folders:
* 'experiments' folder contains codes to train/test Siamese network for performance assessment and DNN for onset detection. 
* 'models' folder contains the codes to implement several different DNN topologies that we experimented.
* 'dataset' folder contains the codes to prepare the files in the database to train the models. 
* 'callback' folder contains the codes to evaluate the performance of the network during the training.

In order to run the codes under experiments directory, features that will be fed into the networks should be saved to a directory. 
The name of this feature directory should be specified in the configuration.py as follows;

* audio_path_trn_per: Database directory for train performance files.
* audio_path_test_per: Database directory for test performance files.
* audio_path_ref_per: : Database directory for reference files.

If onset features are fed into Siamese network, then the features can be obtained using the codes under ../../data_processing.
We also share rhythm_data.pickle file under data ../../data in which onset labels and onset points obtained with the classical onset 
detection method are available. 
You can basically run get_features.py in order to obtain the features that can fed into the Siamese and DNN-based onset detection networks.
get_features.py reads the ../../data/rhythm_data.pickle and prepares the feature files for the DNN-training.

Moreover, Siamese network can use the onset points obtained with experiments/train_onset.py. experiments/train_onset.py implements DNN-based
onset detection in the paper. We train DNN-based onset detection with spectrogram features. Spectrogram can be obtained using compute_spectrogram() method in utils.py.
The output of the DNN-based onset detection network is onset labels. We pre-process the onset labels using the get_binary_labels() method in utils.py.



