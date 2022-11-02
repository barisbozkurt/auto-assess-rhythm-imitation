
# Exemplary automatic classification test

This folder contains code for automatic assessment for rhythmic pattern imitations using Siamese network.

The folder contains the following subfolders:
* 'experiments' folder contains codes to train/test Siamese network for performance assessment and DNN for onset detection. 
* 'models' folder contains DNN topologies we tried for the performance assessment and the onset detection.
* 'dataset' folder contains codes to prepare the files in the database to train the models. 
* 'callback' folder contains codes to evaluate the performance of the network during the training.

In order to run the codes under experiments directory, features that will be used in the model training should be saved to a directory. 
The name of this database directory should be specified in the configuration.py as follows;

* audio_path_trn_per: Database directory for train performance files.
* audio_path_test_per: Database directory for test performance files.
* audio_path_ref_per: : Database directory for reference files.