import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_absolute_error

from configuration import get_config
config = get_config()

class Siamese_Callback(Callback):
    def __init__(self, dataset):
        super(Siamese_Callback, self).__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        preds, label1, label2 = self.task_evaluation()

        if epoch % 50 == 0:
            scr_save_path = '{}/rhythm_siamese_preds_epoch-{}.txt'.format(config.model_save_path, epoch)
            fOut = open(scr_save_path, 'w')
            self.model.save('{}/rhythm_siamese_model_epoch-{}.hdf5'.format(config.model_save_path, epoch))
        else:
            fOut = None
        mae = self.compute_mae(preds, label1, label2, fOut)
        print('MAE for epoch {} is {}'.format(str(epoch),str(mae)))

    def compute_mae(self, preds, label1, label2, fOut = None):
        count = 0
        grds = []
        for p in preds:
            _, grd, _ = label2[count].split(':')
            grd = int(grd)
            grds.append(grd)

            mae = abs(grd-p[0])
            if fOut is not None:
                fOut.writelines(str(p[0]) + ' ' + str(mae) + ' ' + label1[count] + ' ' + label2[count] + '\n')
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
            if end_ind > len(input_1):
                end_ind = len(input_1)
                is_continue = False
            pred.extend(self.model.predict([input_1[begin_ind:end_ind], input_2[begin_ind:end_ind]]))

            begin_ind = begin_ind + batch_size


        return pred,label1,label2