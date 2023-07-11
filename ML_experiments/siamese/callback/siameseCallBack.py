import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_absolute_error
import os


class Siamese_Callback(Callback):
    def __init__(self, dataset, out_folder):
        super(Siamese_Callback, self).__init__()
        self.dataset = dataset
        self.out_folder = out_folder

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if epoch % 10 == 0:
            preds, label1, label2 = self.task_evaluation()

        if epoch % 50 == 0:
            scr_save_path = '{}/siamese_preds_epoch-{}.txt'.format(self.out_folder, epoch)
            f_out = open(scr_save_path, 'w')
            self.model.save('{}/siamese_model_epoch-{}.hdf5'.format(self.out_folder, epoch))
        else:
            f_out = None

        if epoch % 10 == 0:
            mae = self.compute_mae(preds, label1, label2, f_out)
            print('MAE for epoch {} is {}'.format(str(epoch), str(mae)))

    def compute_mae(self, preds, label1, label2, f_out=None):
        count = 0
        grds = []
        for p in preds:
            _, grd, _ = label2[count].split(':')
            grd = int(grd)
            grds.append(grd)

            mae = abs(grd-p[0])
            if f_out is not None:
                f_out.writelines(str(p[0]) + ' ' + str(mae) + ' ' + label1[count] + ' ' + label2[count] + '\n')
            count = count + 1

        mae = mean_absolute_error(preds, grds)
        return mae

    def task_evaluation(self):
        # Get trials
        [input_1, input_2], [label1, label2] = self.dataset.build_verification_trials()
        batch_size = 100
        begin_ind = 0
        pred = []
        is_continue = True
        while is_continue:
            end_ind = begin_ind + batch_size
            if end_ind >= len(input_1):
                end_ind = len(input_1)
                is_continue = False
            pred.extend(self.model.predict([input_1[begin_ind:end_ind], input_2[begin_ind:end_ind]]))

            begin_ind = begin_ind + batch_size

        return pred, label1, label2
