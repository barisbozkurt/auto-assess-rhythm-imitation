from tensorflow.keras import preprocessing
import soundfile as sf
import pandas as pd
import numpy as np
import os
from scipy import signal
import librosa


class Dataset:
    def __init__(self, audio_path_per, audio_path_ref, config, audio_path_test=None):
        self.audio_path_per = audio_path_per
        self.audio_path_ref = audio_path_ref
        self.audio_path_test = audio_path_test
        self.model_name = config.model_name
        self.config = config

        df1 = pd.DataFrame.from_dict(self.index_set(self.audio_path_per))
        df2 = pd.DataFrame.from_dict(self.index_set(self.audio_path_ref))
        self.df = pd.concat([df1, df2], axis=0)
        if audio_path_test is not None:
            df3 = pd.DataFrame.from_dict(self.index_set(self.audio_path_test))
            self.df = pd.concat([self.df, df3], axis=0)

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.assign(id=self.df.index.values)

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_melody = self.df.to_dict()['melodyname']
        self.datasetid_to_grade = self.df.to_dict()['grade']

        print('Finished indexing data. {} usable files found.'.format(len(self)))

    def __getitem__(self, index):
        if '.npy' in self.datasetid_to_filepath[index]:
            instance = np.load(self.datasetid_to_filepath[index]).astype(dtype=float)
        else:
            method = 'librosa'
            if '.wav' in self.datasetid_to_filepath[index]:
                x, srate = sf.read(self.datasetid_to_filepath[index])
            else:
                x, srate = librosa.load(self.datasetid_to_filepath[index], sr=44100)

            if method == 'scipy':
                f, t, S = signal.spectrogram(x, srate, nfft=self.config.nfft, nperseg=self.config.window_size, noverlap=self.config.skip_size)
                S = np.transpose(preprocessing.sequence.pad_sequences(10 * np.log(S +1.e-10), maxlen=self.config.max_len, padding='post'))
                instance = (S - np.mean(S, axis=0)) / np.std(S, axis=0)
            elif method == 'librosa':
                S = librosa.core.stft(y=x, n_fft=2048, win_length=int(self.config.w_size * srate), hop_length=int(self.config.s_size * srate))
                S = preprocessing.sequence.pad_sequences(np.abs(S) ** 2, maxlen=self.config.max_frm, padding='post')
                mel_basis = librosa.filters.mel(sr=srate, n_fft=2048, n_mels=self.config.num_mels)
                instance = np.transpose(np.log10(np.dot(mel_basis, S) + 1e-6))
                #instance = (S - np.mean(S, axis=0)) / np.std(S, axis=0)

        if np.isnan(instance).any():
            raise (TypeError, "input has NAN values")

        melody = self.datasetid_to_melody[index]
        grade = self.datasetid_to_grade[index]
        filepath = self.datasetid_to_filepath[index]

        label = melody + ':' + str(grade) + ':' +  filepath

        return instance, label

    def __len__(self):
        return len(self.df)

    def index_set(self, audio_path):
        audio_files = []
        print('Indexing dataset')

        for root, folders, files in os.walk(audio_path):
            if len(files) == 0:
                continue

            for f in files:
                # Skip non-sound files
                if not (f.endswith(self.config.file_ext)):
                    continue

                file_path = os.path.join(root, f).replace('\\', '/')
                parts = f.replace(self.config.file_ext, '').split('_')
                melody_name = parts[0] + '_' + parts[1]
                if '_per' in f:
                    grade = int(parts[4].replace('grade', ''))
                    result = parts[3]
                    i_type = 'per'
                else:
                    grade = int(parts[3].replace('grade', ''))
                    result = 'None'
                    i_type = 'ref'

                    if grade != 4:
                        continue

                onset_file_path = file_path.replace(self.config.file_ext, self.config.lab_ext)
                if os.path.isfile(onset_file_path):
                    labels = np.loadtxt(onset_file_path, dtype=float)
                else:
                    labels = None

                audio_files.append({
                    'melodyname': melody_name,
                    'filepath': file_path,
                    'result': result,
                    'type': i_type,
                    'grade': grade,
                    'label': labels
                })

        return audio_files


