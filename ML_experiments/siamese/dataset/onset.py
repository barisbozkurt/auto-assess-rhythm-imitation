import numpy as np
from dataset import Dataset

class Dataset_Onset(Dataset):
    def __init__(self, audio_path_per, audio_path_ref, config, audio_path_test=None):
        super().__init__(audio_path_per, audio_path_ref, config, audio_path_test)

    def yield_batches(self, batchsize=None):
        while True:
            (input, labels) = self.build_batch(batchsize)
            yield input, labels

    def build_verification_trials(self):
        samples = self.df

        labels = []
        for kk in range(len(samples)):
            labels.append(samples['label'].values[kk])
        outputs = np.array(labels)[:, :, np.newaxis]
        inputs = list(zip(samples['id'].values))

        in1 = [self[i] for i in list(zip(*inputs))[0]]
        input_1, input_1_label = np.stack(list(zip(*in1))[0]), np.stack(list(zip(*in1))[1])

        #input_1, outputs = self.partition_samples_test(input_1, outputs, frm_size=frmsize)

        return input_1, outputs, input_1_label

    def build_batch(self, batchsize):
        samples = self.df.sample(int(batchsize))

        labels = []
        for kk in range(len(samples)):
            labels.append(samples['label'].values[kk])
        outputs = np.array(labels)[:, :, np.newaxis]
        inputs = list(zip(samples['id'].values))

        in1 = [self[i] for i in list(zip(*inputs))[0]]
        input_1 = np.stack(list(zip(*in1))[0])

        #input_1, outputs = self.partition_samples(input_1, outputs, batchsize, frmsize)

        return input_1, outputs

