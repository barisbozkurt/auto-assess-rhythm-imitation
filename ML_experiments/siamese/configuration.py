import argparse

parser = argparse.ArgumentParser()    # make parser

def new_method():
    return 0

# get arguments
def get_config():
    """

    :rtype: object
    """
    config, unparsed = parser.parse_known_args()
    return config

# return bool type of argument
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Database and feature extraction parameters
data_arg = parser.add_argument_group('Dataset')
data_arg.add_argument('--audio_path_trn_per', type=str, default='../../data/feat/train', help="audio directory path for student train performances")
data_arg.add_argument('--audio_path_test_per', type=str, default='../../data/feat/test', help="audio directory path for student test performances")
data_arg.add_argument('--audio_path_ref', type=str, default='../../data/feat/ref', help="audio directory path for references")
data_arg.add_argument('--model_save_path', type=str, default='../../data/models', help="model directory path to save models")
data_arg.add_argument('--w_size', type=float, default=0.025, help="window length (in sec)")
data_arg.add_argument('--s_size', type=float, default=0.025, help="skip size (in sec)")
data_arg.add_argument('--max_frm', type=int, default=320, help="number of frames after padding (num_frame * skip_size = max duration in sec)")
)data_arg.add_argument('--num_feat', type=int, default=20, help="dimension of the features") # 64 if the input is spectrogram
data_arg.add_argument('--file_ext', type=str, default='_onset_CNN1D+LSTM.npy', help="extension for feature file in Siamese network")
data_arg.add_argument('--lab_ext', type=str, default='none', help="extension for onset labels") # keep that 'none' in Siamese network, give the extension of onset labels when training the DNN-based onset detection

# Model parameters
model_arg = parser.add_argument_group('Model')
model_arg.add_argument('--model_name', type=str, default='FF', help="model type")
model_arg.add_argument('--batch_size', type=int, default=16, help="batch size")
model_arg.add_argument('--embedding_dimension', type=int, default=16, help="embedding dimension")
model_arg.add_argument('--num_filter', type=int, default=24, help="number of filters in CNN")
model_arg.add_argument('--num_units', type=int, default=36, help="number of units in RNN")
model_arg.add_argument('--epochs', type=int, default=200, help="number of epochs")
model_arg.add_argument('--steps_per_epoch', type=int, default=200, help="number of iterations")
model_arg.add_argument('--dropout', type=float, default=0.05, help="dropout rate")

config = get_config()
print(config)           # print all the arguments
