import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_absolute_error,precision_score, recall_score, f1_score, confusion_matrix
from utils import compute_metrics
import os


class Onset_Callback(Callback):
    def __init__(self, dataset, out_folder):
        super(Onset_Callback, self).__init__()
        self.dataset = dataset
        self.out_folder = out_folder

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        preds, labels, preds_path = self.task_evaluation()

        onset_ref_preds = []
        onset_ref_labels = []
        onset_test_preds = []
        onset_test_labels = []
        onset_train_preds = []
        onset_train_labels = []

        count = 0
        for path in preds_path:
            if '/ref/' in path:
                onset_ref_preds.append(preds[count])
                onset_ref_labels.append(labels[count])
            if '/train/' in path:
                onset_train_preds.append(preds[count])
                onset_train_labels.append(labels[count])
            if '/test/' in path:
                onset_test_preds.append(preds[count])
                onset_test_labels.append(labels[count])

            if epoch % 50 == 0 and epoch > 0:
                _, _, file_path = path.split(':')
                out_path = file_path.replace('_spec.npy', '_cnn1D_onset_epoch{}.txt'.format(epoch))
                np.savetxt(out_path, preds[count].astype(float), fmt='%.1f')

            count = count + 1

        print('Epoch {}'.format(str(epoch)))

        metric_base_ref, metric_mod_ref = compute_metrics(np.array(onset_ref_preds), np.array(onset_ref_labels))
        print('Ref (metrics base): recall {}, precision {}, F1 {}'.format(str(metric_base_ref[0]),
                                                                                      str(metric_base_ref[1]),
                                                                                      str(metric_base_ref[2])))

        print('Ref (metrics mod): recall {}, precision {}, F1 {}, accuracy {}'.format(str(metric_mod_ref[0]),
                                                                                                 str(metric_mod_ref[1]),
                                                                                                 str(metric_mod_ref[2]),
                                                                                                 str(metric_mod_ref[3])))

        metric_base_train, metric_mod_train = compute_metrics(np.array(onset_train_preds),np.array(onset_train_labels))
        print('Train (metrics base): recall {}, precision {}, F1 {}'.format(str(metric_base_train[0]),
                                                                          str(metric_base_train[1]),
                                                                          str(metric_base_train[2])))

        print('Train (metrics mod): recall {}, precision {}, F1 {}, accuracy {}'.format(str(metric_mod_train[0]),
                                                                                      str(metric_mod_train[1]),
                                                                                      str(metric_mod_train[2]),
                                                                                      str(metric_mod_train[3])))

        metric_base_test, metric_mod_test = compute_metrics(np.array(onset_test_preds),np.array(onset_test_labels))
        print('Test (metrics base): recall {}, precision {}, F1 {}'.format(str(metric_base_test[0]),
                                                                          str(metric_base_test[1]),
                                                                          str(metric_base_test[2])))

        print('Test (metrics mod): recall {}, precision {}, F1 {}, accuracy {}'.format(str(metric_mod_test[0]),
                                                                                       str(metric_mod_test[1]),
                                                                                       str(metric_mod_test[2]),
                                                                                       str(metric_mod_test[3])))

        if epoch % 50 == 0 and epoch > 0:
            f_out = open('{}/onset_preds_epoch-{}.txt'.format(self.out_folder, epoch), 'w')
            f_out.writelines('Epoch {}: recall {}, precision {}, F1 score {}, sample accuracy {}'.format(
                str(epoch), str(metric_mod_test[0]), str(metric_mod_test[1]), str(metric_mod_test[2]), str(metric_mod_test[3])))
            self.model.save('{}/onset_model_epoch-{}.hdf5'.format(self.out_folder,epoch))

            print('Model is saved')

    def task_evaluation(self):
        # Get trials
        [input_1, label, input_1_path] = self.dataset.build_verification_trials()
        batch_size = 64
        begin_ind = 0
        end_ind = begin_ind + batch_size
        preds = []
        is_continue = True
        while is_continue:
            if end_ind >= len(input_1):
                end_ind = len(input_1)
                is_continue = False
            preds.extend(self.model.predict(input_1[begin_ind:end_ind,:,:]))
            begin_ind = begin_ind + batch_size
            end_ind = begin_ind + batch_size

        label = np.reshape(label,(-1,config.max_frm))
        preds = np.reshape(np.array(preds),(-1,config.max_frm))
        return preds, label, input_1_path
