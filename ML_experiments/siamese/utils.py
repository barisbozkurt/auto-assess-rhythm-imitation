import numpy as np
import librosa
from tensorflow.keras import preprocessing
import matplotlib.pyplot as plt
import librosa.display
from sklearn.metrics import mean_absolute_error,precision_score, recall_score, f1_score, confusion_matrix

num_max_frm = 320
num_max_onsets = 20
num_max_onsets_simple = 50
skip_size = 0.025
window_size = 0.025

def compute_metrics_base(preds, labels):
    preds_flat = preds.ravel()
    labels_flat = labels.ravel()

    _, _, _, _, tn, fp, _, fn, tp = confusion_matrix(labels_flat, preds_flat).ravel()

    if tp + fn == 0:
        recall = -1
    else:
        recall = float(tp) / float(tp + fn)

    if tp + fp == 0:
        precision = -1
    else:
        precision = float(tp) / float(tp + fp)

    if recall + precision == 0:
        F1 = -1
    else:
        F1 = float(2 * recall * precision) / float(recall + precision)

    return recall, precision, F1


def compute_metrics_mod(preds, labels):
    fn = 0
    fp = 0
    tp = 0
    tn = 0
    t_sample = 0
    for c_sample in range(labels.shape[0]):
        is_t_sample = True
        for c_index in range(labels.shape[1]):
            if labels[c_sample, c_index] == -1:
                break

            if c_index < 2:
                begin_ind = 0
            else:
                begin_ind = c_index - 2

            if c_index >= labels.shape[1] - 2:
                end_ind = labels.shape[1]
            else:
                end_ind = c_index + 3

            if labels[c_sample, c_index] == 1:
                if 1 in preds[c_sample, begin_ind:end_ind]:
                    tp = tp + 1
                else:
                    is_t_sample = False
                    fn = fn + 1
            else:
                if not 1 in preds[c_sample, begin_ind:end_ind]:
                    tn = tn + 1

            if preds[c_sample, c_index] == 1:
                if not 1 in labels[c_sample, begin_ind:end_ind]:
                    is_t_sample = False
                    fp = fp + 1

        if is_t_sample:
            t_sample = t_sample + 1

    if tp + fn == 0:
        recall = -1
    else:
        recall = float(tp) / float(tp + fn)

    if tp + fp == 0:
        precision = -1
    else:
        precision = tp / (tp + fp)

    if recall + precision == 0:
        F1 = -1
    else:
        F1 = (2 * recall * precision) / (recall + precision)

    acc_sample = 100 * float(t_sample) / float(labels.shape[0])
    all_counts = [tn, fp, fn, tp]

    return recall, precision, F1, acc_sample, all_counts


def preprocess_labels(labels):
    labels_proc = labels.copy()

    labels_proc[np.where((labels_proc > -1) & (labels_proc < 1))] = 0
    labels_proc[np.where(labels_proc == 1)] = 1

    return labels_proc

def preprocess_preds_simple(preds):
    preds_proc = preds.copy()

    preds_proc[np.where((preds_proc < 0.9))] = 0
    preds_proc[np.where(preds_proc >= 0.9)] = 1

    return preds_proc

def preprocess_preds(preds, threshold = 0.7):
    preds_proc = np.zeros((preds.shape[0], preds.shape[1]))

    for c_sample in range(preds.shape[0]):
        for c_index in range(0, preds.shape[1]):
            if c_index == 0:
                b_ind_frm = c_index
                e_ind_frm = c_index + 2
            elif c_index == preds.shape[1] - 1:
                b_ind_frm = c_index - 1
                e_ind_frm = c_index + 1
            else:
                b_ind_frm = c_index - 1
                e_ind_frm = c_index + 2

            mean_ind = np.mean(preds[c_sample, b_ind_frm:e_ind_frm])
            if e_ind_frm < preds.shape[1]:
                mean_ind_next = np.mean(preds[c_sample, (b_ind_frm+1):(e_ind_frm+1)])
            else:
                mean_ind_next = 0

            if mean_ind >= threshold and mean_ind >= mean_ind_next:
                is_insert_one = True
                if c_index == 0:
                    b_check_ind = -1
                elif c_index <= 3:
                    b_check_ind = 0
                else:
                    b_check_ind = c_index - 3

                if b_check_ind >= 0:
                    if max(preds_proc[c_sample, b_check_ind:c_index]) == 1:
                        is_insert_one = False

                if is_insert_one:
                    preds_proc[c_sample, c_index] = 1

    return preds_proc

def compute_metrics(preds, labels):
    labels_proc = preprocess_labels(labels)
    preds_proc = preprocess_preds(preds)
    #preds_proc_simple = preprocess_preds_simple(preds)

    recall, precision, F1 = compute_metrics_base(preds_proc, labels_proc)
    recall_mod, precision_mod, F1_mod, acc_sample, all_counts = compute_metrics_mod(preds_proc, labels_proc)

    metric_base = [recall, precision, F1]
    metric_mod = [recall_mod, precision_mod, F1_mod, acc_sample, all_counts]

    return metric_base, metric_mod

def compute_metrics_singleFile(preds, labels):
    labels_proc = preprocess_labels(labels)
    preds_proc = preprocess_preds(preds)
    #preds_proc_simple = preprocess_preds_simple(preds)

    recall_mod, precision_mod, F1_mod, acc_sample, counts = compute_metrics_mod(preds_proc, labels_proc)

    metric_mod = [recall_mod, precision_mod, F1_mod, acc_sample, counts]

    return metric_mod, np.where(labels_proc == 1), np.where(preds_proc == 1)


def get_binary_labels(onsets, num_frm=None):
    onsets_lab = np.zeros([num_max_frm, 1])
    if num_frm != None:
        onsets_lab[(num_frm+1):len(onsets_lab)] = -1

    c_frm = 0
    onset_ind = 0
    start_ind = 0
    end_ind = window_size
    while c_frm < num_max_frm and onset_ind < len(onsets):
        if start_ind < onsets[onset_ind] <= end_ind:
            if c_frm == 0:
                onsets_lab[c_frm] = 1
                onsets_lab[c_frm + 1] = 0.9
                onsets_lab[c_frm + 2] = 0.6
            elif c_frm == 1:
                onsets_lab[c_frm-1] = 0.9
                onsets_lab[c_frm] = 1
                onsets_lab[c_frm + 1] = 0.9
                onsets_lab[c_frm + 2] = 0.6
            elif c_frm == num_frm - 2:
                onsets_lab[c_frm - 2] = 0.6
                onsets_lab[c_frm - 1] = 0.9
                onsets_lab[c_frm] = 1
                onsets_lab[c_frm + 1] = 0.9
            elif c_frm == num_frm - 1:
                onsets_lab[c_frm - 2] = 0.6
                onsets_lab[c_frm - 1] = 0.9
                onsets_lab[c_frm] = 1
            else:
                onsets_lab[c_frm-2] = 0.6
                onsets_lab[c_frm-1] = 0.9
                onsets_lab[c_frm] = 1
                onsets_lab[c_frm + 1] = 0.9
                onsets_lab[c_frm + 2] = 0.6

            onset_ind = onset_ind + 1

        start_ind = start_ind + skip_size
        end_ind = end_ind + skip_size

        c_frm = c_frm + 1

    return onsets_lab


def compute_spectrogram(wav_file_path):
    x, sr = librosa.load(wav_file_path, sr=44100)
    #S = librosa.feature.mfcc(y=x, sr=sr, n_mfcc = 32, hop_length=int(skip_size*sr),win_length=int(window_size*sr),)

    S = librosa.feature.melspectrogram(y=x, sr=sr,
                                       n_fft=2048,
                                       n_mels=64,
                                       #fmin = 100,
                                       #fmax = 32000,
                                       hop_length=int(skip_size*sr),
                                       win_length=int(window_size*sr),
                                       power=2.0)

    num_frm = S.shape[1]
    S = preprocessing.sequence.pad_sequences(S, maxlen=num_max_frm, padding='post', dtype=float)
    S = np.transpose(S)
    S = np.array(librosa.power_to_db(S + (1e-6 * np.random.randn(S.shape[0], S.shape[1]))))

    if (np.where(np.isnan(S)))[0].shape[0]:
        print('Spectrogram contains NAN values')

    if False:
        fig, ax = plt.subplots()
        librosa.display.specshow(np.transpose(S), x_axis='time', y_axis='mel', sr=sr, fmax=22000, ax=ax)
        ax.set(title='Mel-frequency spectrogram')

    return S, num_frm


def read_label(label_file_path):
    fin = open(label_file_path, "r")

    onsets = []
    for line in fin:
        onset, _, _ = line.split('\t')
        onsets.append(float(onset))
    return np.array(onsets)

def get_num_max_onset_simple():
    return num_max_onsets_simple

def get_num_max_onset():
    return num_max_onsets

def get_num_max_frm():
    return num_max_frm

def get_ws():
    return window_size

def get_ss():
    return skip_size
