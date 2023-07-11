import pandas as pd
import numpy as np
import random
from dataset import Dataset

class Dataset_Siamese(Dataset):
    def __init__(self, audio_path_per, audio_path_ref, config):
        super().__init__(audio_path_per, audio_path_ref, config, audio_path_test=None)

    def get_pairs(self):
        """Generates a list of 2-tuples containing pairs of dataset IDs belonging to the same speaker."""
        uniq_mels = list(set(self.df['melodyname'].values))

        pairs = pd.DataFrame()
        for melody in uniq_mels:
            sample_ref = self.df[
                (self.df['melodyname'] == melody) & (self.df['type'] == 'ref')]

            sample_per = self.df[
                (self.df['melodyname'] == melody) & (self.df['type'] == 'per')]

            sample = (pd.merge(sample_ref, sample_per, on='melodyname'))

            #pairs = pairs.append(sample)
            pairs = pd.concat([pairs, sample])
        pairs = list(zip(pairs['id_x'].values, pairs['id_y'].values))

        return pairs

    def get_random_pairs(self, num_pairs):
        """Generates a list of 2-tuples containing pairs of dataset IDs belonging to the same speaker."""
        num_melody = int(num_pairs / 1) # I will choose this much melody for every batch
        df_per = self.df[self.df['type'] == 'per']
        melodies = random.sample(list(set(df_per['melodyname'].values)), num_melody)

        pairs = pd.DataFrame()
        labels = []
        for melody in melodies:
            sample_ref = self.df[
                (self.df['melodyname'] == melody) & (self.df['type'] == 'ref')].sample(1)

            sample_per = self.df[
                (self.df['melodyname'] == melody) & (self.df['type'] == 'per')].sample(1)

            sample = (pd.merge(sample_ref, sample_per, on='melodyname'))

            labels.extend(sample['grade_y'].values)
            #pairs = pairs.append(sample)
            pairs = pd.concat([pairs, sample])

        pairs = list(zip(pairs['id_x'].values, pairs['id_y'].values))
        outputs = np.array(labels)[:, np.newaxis]

        return pairs, outputs

    def get_random_pairs_balanced(self, num_pairs):
        """Generates a list of 2-tuples containing pairs of dataset IDs belonging to the same speaker."""
        num_melody = int(num_pairs / 1) # I will choose this much melody for every batch
        df_per = self.df[self.df['type'] == 'per']
        try:
            melodies = random.sample(list(set(df_per['melodyname'].values)), num_melody)
        except:
            aa = 9

        pairs = pd.DataFrame()
        labels = []
        c_grd = 0
        for melody in melodies:
            m_grd = c_grd % 4 + 1
            sample_ref = self.df[
                (self.df['melodyname'] == melody) & (self.df['type'] == 'ref')].sample(1)

            if self.df[
                (self.df['melodyname'] == melody) & (self.df['type'] == 'per') & (self.df['grade'] == m_grd)].shape[0] != 0:

                sample_per = self.df[
                    (self.df['melodyname'] == melody) & (self.df['type'] == 'per') & (self.df['grade'] == m_grd)].sample(1)
            else:
                sample_per = self.df[
                    (self.df['melodyname'] == melody) & (self.df['type'] == 'per')].sample(1)

            sample = (pd.merge(sample_ref, sample_per, on='melodyname'))

            labels.extend(sample['grade_y'].values)
            #pairs = pairs.append(sample)
            pairs = pd.concat([pairs, sample])

            c_grd = c_grd + 1

        pairs = list(zip(pairs['id_x'].values, pairs['id_y'].values))
        outputs = np.array(labels)[:, np.newaxis]

        return pairs, outputs

    def build_batch(self, batchsize):
        """
        This method builds a batch of verification task samples meant to be input into a siamese network. Each sample
        is two instances of the dataset retrieved with the __getitem__ function and a label which indicates whether the
        instances belong to the same speaker or not. Each batch is 50% pairs of instances from the same speaker and 50%
        pairs of instances from different speakers.
        :param batchsize: Number of verification task samples to build the batch out of.
        :return: Inputs for both sides of the siamese network and outputs indicating whether they are from the same
        speaker or not.
        """
        #pairs, outputs = self.get_random_pairs(batchsize)
        pairs, outputs = self.get_random_pairs_balanced(batchsize)

        # Take the instances and labels and stack to form a batch of pairs of instances from the same melody
        in1 = [self[i] for i in list(zip(*pairs))[0]]
        in2 = [self[i] for i in list(zip(*pairs))[1]]

        input_1 = np.stack(list(zip(*in1))[0])
        input_2 = np.stack(list(zip(*in2))[0])

        if self.model_name == 'CNN2D':
            input_1, input_2 = input_1[:, :, :, np.newaxis], input_2[:, :, :, np.newaxis]

        return [input_1, input_2], outputs

    def build_verification_trials(self):
        """
        This method builds a batch of verification task samples meant to be input into a siamese network. Each sample
        is two instances of the dataset retrieved with the __getitem__ function and a label which indicates whether the
        instances belong to the same speaker or not. Each batch is 50% pairs of instances from the same speaker and 50%
        pairs of instances from different speakers.
        :param batchsize: Number of verification task samples to build the batch out of.
        :return: Inputs for both sides of the siamese network and outputs indicating whether they are from the same
        speaker or not.
        """
        pairs = self.get_pairs()

        # Take the instances and labels and stack to form a batch of pairs of instances from the same melody
        in1 = [self[i] for i in list(zip(*pairs))[0]]
        in2 = [self[i] for i in list(zip(*pairs))[1]]

        input_1, input_1_label = np.stack(list(zip(*in1))[0]), np.stack(list(zip(*in1))[1])
        input_2, input_2_label = np.stack(list(zip(*in2))[0]), np.stack(list(zip(*in2))[1])

        if self.model_name == 'CNN2D':
            input_1, input_2 = input_1[:, :, :, np.newaxis],input_2[:, :, :, np.newaxis]

        return [input_1, input_2], [input_1_label, input_2_label]

    def yield_batches(self, batchsize=None):
        """Convenience function to yield verification batches forever."""
        if batchsize is None:
            ([input_1, input_2], [input_1_label, input_2_label]) = self.build_verification_trials()
            yield ([input_1, input_2], [input_1_label, input_2_label])
        else:
            while True:
                ([input_1, input_2], labels) = self.build_batch(batchsize)
                yield ([input_1, input_2], labels)

