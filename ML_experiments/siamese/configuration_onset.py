import argparse

parser = argparse.ArgumentParser()    # make parser

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
data_arg.add_argument('--audio_path_trn_per', type=str, default='../../../data/feat/train', help="audio directory path for train")
data_arg.add_argument('--audio_path_test_per', type=str, default='../../../data/feat/test', help="audio directory path for test")
data_arg.add_argument('--audio_path_ref', type=str, default='../../../data/feat/ref', help="audio directory path for references")
data_arg.add_argument('--model_save_path', type=str, default='../../../data/output/models', help="audio directory path for test")
data_arg.add_argument('--log_save_path', type=str, default='../../../data/output/logs', help="audio directory path for test")
data_arg.add_argument('--w_size', type=float, default=0.025, help="window length (in sec)")
data_arg.add_argument('--s_size', type=float, default=0.025, help="skip size (in sec)")
data_arg.add_argument('--max_frm', type=int, default=320, help="number of frames after padding (num_frame * skip_size = max duration (in sec))")
data_arg.add_argument('--num_frm_split', type=int, default=320, help="the number of frames in each chunk in a split")
data_arg.add_argument('--num_feat', type=int, default=64, help="dimension of the features")
data_arg.add_argument('--file_ext', type=str, default='_spec.npy', help="extension for spectrogram file")
data_arg.add_argument('--lab_ext', type=str, default='_onset_lab_bin.txt', help="extension for onset labels")

# Model parameters
model_arg = parser.add_argument_group('Model')
model_arg.add_argument('--model_name', type=str, default='CNN1D_RNN_seq2seq', help="model type")
model_arg.add_argument('--batch_size', type=int, default=64, help="batch size")
model_arg.add_argument('--embedding_dimension', type=int, default=16, help="embedding dimension")
model_arg.add_argument('--num_filter', type=int, default=24, help="number of filters in cnn")
model_arg.add_argument('--num_units', type=int, default=36, help="number of units in rnn")
model_arg.add_argument('--epochs', type=int, default=250, help="max iteration")
model_arg.add_argument('--steps_per_epoch', type=int, default=200, help="max iteration")
model_arg.add_argument('--dropout', type=float, default=0.05, help="max iteration")

config = get_config()
print(config)           # print all the arguments
