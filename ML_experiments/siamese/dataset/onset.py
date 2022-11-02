import numpy as np
from dataset import Dataset

class Dataset_Onset(Dataset):
    def __init__(self, audio_path_per, audio_path_ref, model_name, audio_path_test = None):
        super().__init__(audio_path_per, audio_path_ref, model_name, audio_path_test)

    def yield_batches(self, batchsize=None, frmsize=None):
        while True:
            (input, labels) = self.build_batch(batchsize, frmsize)
            yield input, labels

    """
    def partition_samples_test(self, feats, labels, frm_size):
        samples = np.reshape(feats, (-1, frm_size, feats.shape[2]))
        sample_labels = np.reshape(labels, (-1, frm_size))

        return np.array(samples), np.array(sample_labels)

    def partition_samples(self, feats, labels, batch_size, frm_size):
        c_sample = 0
        samples = []
        sample_labels = []
        is_last = False
        for kk in range(feats.shape[0]):
            count = 0
            is_continue = True
            while is_continue:
                first_frm = count * frm_size
                last_frm = (count + 1) * frm_size

                sample = feats[kk, first_frm:last_frm, :]
                sample_label = labels[kk, first_frm:last_frm]

                if -1 in sample_label:
                    sample_label[np.where(sample_label < 0)] = 0
                    is_continue = False

                if max(sample_label) != 0:
                    samples.append(sample)
                    sample_labels.append(sample_label)

                    c_sample = c_sample + 1
                count = count + 1

                if c_sample == batch_size:
                    is_last = True
                    break

            if is_last:
                break
        return np.array(samples), np.array(sample_labels)

    """

    def build_verification_trials(self, frmsize):
        """
        This method builds a batch of verification task samples meant to be input into a siamese network. Each sample
        is two instances of the dataset retrieved with the __getitem__ function and a label which indicates whether the
        instances belong to the same speaker or not. Each batch is 50% pairs of instances from the same speaker and 50%
        pairs of instances from different speakers.
        :param batchsize: Number of verification task samples to build the batch out of.
        :return: Inputs for both sides of the siamese network and outputs indicating whether they are from the same
        speaker or not.
        """
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

    def build_batch(self, batchsize, frmsize):
        """
        This method builds a batch of verification task samples meant to be input into a siamese network. Each sample
        is two instances of the dataset retrieved with the __getitem__ function and a label which indicates whether the
        instances belong to the same speaker or not. Each batch is 50% pairs of instances from the same speaker and 50%
        pairs of instances from different speakers.
        :param batchsize: Number of verification task samples to build the batch out of.
        :return: Inputs for both sides of the siamese network and outputs indicating whether they are from the same
        speaker or not.
        """
        samples = self.df.sample(int(batchsize))

        labels = []
        for kk in range(len(samples)):
            labels.append(samples['label'].values[kk])
        outputs = np.array(labels)[:,:,np.newaxis]
        inputs = list(zip(samples['id'].values))

        in1 = [self[i] for i in list(zip(*inputs))[0]]
        input_1 = np.stack(list(zip(*in1))[0])

        #input_1, outputs = self.partition_samples(input_1, outputs, batchsize, frmsize)

        return input_1, outputs

