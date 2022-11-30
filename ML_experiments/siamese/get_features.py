# some_file.py
import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../../data_processing/')

import os
import pickle
from datapreprocess_rhythm import Recording
import numpy as np
from os import path
from utils import get_binary_labels,compute_metrics,compute_spectrogram,read_label,get_num_max_frm,get_num_max_onset

def save_all_feats(rec, file_path_init):
    # ref_bin_onset_vect = ref_rec.binary_onset_vector[onset_method]
    onsets_base = rec.onsets[onset_method]
    onsets_base_pad = np.concatenate((onsets_base, np.zeros(get_num_max_onset() - len(onsets_base), dtype=float)), axis=0)

    wav_file_path = '{}/{}'.format(in_wav_folder, rec.file_path.replace('.m4a', '.wav'))
    [S, num_frm] = compute_spectrogram(wav_file_path)

    onsets_base_bin = get_binary_labels(onsets_base, num_frm)

    out_file_path = '{}_{}'.format(file_path_init, 'spec.npy')
    np.save(out_file_path, S)

    # file_path = '{}_{}'.format(file_path_init, 'bin_onset.npy')
    # np.save(file_path, ref_bin_onset_vect)

    out_file_path = '{}_{}'.format(file_path_init, 'onset_base.npy')
    np.save(out_file_path, onsets_base_pad)

    if path.exists(in_lab_folder + '/' + rec.file_path.replace('.m4a', '.txt')):
        onset_labels = read_label(in_lab_folder + '/' + rec.file_path.replace('.m4a', '.txt'))
    else:
        onset_labels = onsets_base

    onset_labels_pad = np.concatenate((onset_labels, np.zeros(get_num_max_onset() - len(onset_labels), dtype=float)), axis=0)

    out_file_path = '{}_{}'.format(file_path_init, 'onset_lab.npy')
    np.save(out_file_path, onset_labels_pad)

    onset_labels_bin = get_binary_labels(onset_labels, num_frm)
    out_file_path = '{}_{}'.format(file_path_init, 'onset_lab_bin.txt')
    np.savetxt(out_file_path, onset_labels_bin.astype(float), fmt='%.1f')

    return onsets_base_bin, onset_labels_bin, num_frm

if __name__ == "__main__":
    in_wav_folder = '../../data/wav/rhythm'
    in_lab_folder = '../../data/labels/manuallyCorrectedOnsets'
    feat_folder = '../../data/feat/rhythm'
    out_result_path = '../../data/output/results_onset_base.txt'

    if not os.path.exists(feat_folder + '/ref'):
        os.makedirs(feat_folder + '/ref')
    if not os.path.exists(feat_folder + '/test'):
        os.makedirs(feat_folder + '/test')
    if not os.path.exists(feat_folder + '/train'):
        os.makedirs(feat_folder + '/train')

    test_file_list_file = '../../data/listperformances_test.txt'
    temp_data = np.loadtxt(test_file_list_file, dtype={'names': ('file', 'grade'), 'formats': ('S30', 'i4')})
    test_files = {file.decode("utf-8"): grade for file, grade in temp_data}

    # Reading and accessing features
    data_file_name = '../../data/rhythm_data.pickle'
    onset_method = 'hfc'
    #reading pickle
    with open(data_file_name, 'rb') as handle:
        rhy_data_read = pickle.load(handle)

        onset_ref_preds = []
        onset_ref_labels = []
        onset_test_preds = []
        onset_test_labels = []
        onset_train_preds = []
        onset_train_labels = []

        for exercise in rhy_data_read.keys():
            for ref_rec in rhy_data_read[exercise]['ref']:
                file_path_init = '{}/ref/{}_grade{}'.format(feat_folder,ref_rec.file_path.replace('.m4a', ''), ref_rec.grade)
                onsets_base_bin, onset_labels_bin, num_frm = save_all_feats(ref_rec, file_path_init)
                onset_ref_labels.append(onset_labels_bin)
                onset_ref_preds.append(onsets_base_bin)
            for per_rec in rhy_data_read[exercise]['per']:
                if per_rec.file_path.replace('.m4a', '.wav') in test_files:  # put data to test set
                    file_path_init = '{}/test/{}_grade{}'.format(feat_folder,per_rec.file_path.replace('.m4a',''), per_rec.grade)
                    onsets_base_bin, onset_labels_bin, num_frm = save_all_feats(per_rec, file_path_init)
                    onset_test_labels.append(onset_labels_bin)
                    onset_test_preds.append(onsets_base_bin)
                else:  # put data to train set
                    file_path_init = '{}/train/{}_grade{}'.format(feat_folder,per_rec.file_path.replace('.m4a', ''), per_rec.grade)
                    onsets_base_bin, onset_labels_bin, num_frm = save_all_feats(per_rec, file_path_init)
                    onset_train_labels.append(onset_labels_bin)
                    onset_train_preds.append(onsets_base_bin)

    metric_base_ref, metric_mod_ref = compute_metrics(np.squeeze(np.array(onset_ref_preds)), np.squeeze(np.array(onset_ref_labels)))
    metric_base_test, metric_mod_test = compute_metrics(np.squeeze(np.array(onset_test_preds)), np.squeeze(np.array(onset_test_labels)))
    metric_base_train, metric_mod_train = compute_metrics(np.squeeze(np.array(onset_train_preds)), np.squeeze(np.array(onset_train_labels)))

    fout = open(out_result_path, "w")
    str_prn = ['Reference', 'Test', 'Train']
    fout.writelines('BASE METRICS\n')
    for kk in range(3):
        if kk == 0:
            recall, precision, F1 = metric_base_ref
        if kk == 1:
            recall, precision, F1 = metric_base_test
        if kk == 2:
            recall, precision, F1 = metric_base_train

        fout.writelines('{}, Recall: {}, Precision: {}, F1: {}\n'.format(str_prn[kk], recall, precision, F1))

    fout.writelines('MODIFIED  METRICS\n')
    for kk in range(3):
        if kk == 0:
            recall, precision, F1, acc, counts = metric_mod_ref
        if kk == 1:
            recall, precision, F1, acc, counts = metric_mod_test
        if kk == 2:
            recall, precision, F1, acc, counts = metric_mod_train

        fout.writelines('{}, Recall: {}, Precision: {} F1: {}, Acc: {}\n'.format(str_prn[kk], recall, precision, F1, acc))
        fout.writelines('TN: {}, FP: {}, FN: {}, TP: {}\n'.format(counts[0], counts[1], counts[2], counts[3]))