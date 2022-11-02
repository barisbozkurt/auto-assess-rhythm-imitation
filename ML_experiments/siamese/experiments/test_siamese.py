from tensorflow import keras
from configuration import get_config
import numpy as np

config = get_config()

model_path = '../models/model_epoch-299.hdf5'

model = keras.models.load_model(model_path)
input_1 = np.loadtxt('../ornekler/pair1_mae2.89/59_rhy1_per131160_pass_grade4_onsets.txt')
input_2 = np.loadtxt('../ornekler/pair1_mae2.89/59_rhy1_ref340759_grade4_onsets.txt')

input_1 = input_1[np.newaxis,:]
input_2 = input_2[np.newaxis,:]

pred = model.predict([input_1, input_2])
mae = abs(pred-4)

input_1_corr = np.loadtxt('../ornekler/pair1_mae2.89/59_rhy1_per131160_pass_grade4_onsets_corrected.txt')
input_2_corr = np.loadtxt('../ornekler/pair1_mae2.89/59_rhy1_ref340759_grade4_onsets_corrected.txt')

input_1_corr = input_1_corr[np.newaxis,:]
input_2_corr = input_2_corr[np.newaxis,:]

pred_corr = model.predict([input_1_corr, input_2_corr])
mae_corr = abs(pred_corr-4)

print('Original MAE {} Corrected MAE {}'.format(str(mae[0]), str(mae_corr[0])))


input_1 = np.loadtxt('../ornekler/pair2_mae2.88/67_rhy1_per1939742_pass_grade4_onsets.txt')
input_2 = np.loadtxt('../ornekler/pair2_mae2.88/67_rhy1_ref2838732_grade4_onsets.txt')

input_1 = input_1[np.newaxis,:]
input_2 = input_2[np.newaxis,:]

pred = model.predict([input_1, input_2])
mae = abs(pred-4)

input_1_corr = np.loadtxt('../ornekler/pair2_mae2.88/67_rhy1_per1939742_pass_grade4_onsets_corrected.txt')
input_2_corr = np.loadtxt('../ornekler/pair2_mae2.88/67_rhy1_ref2838732_grade4_onsets_corrected.txt')

input_1_corr = input_1_corr[np.newaxis,:]
input_2_corr = input_2_corr[np.newaxis,:]

pred_corr = model.predict([input_1_corr, input_2_corr])
mae_corr = abs(pred_corr-4)

print('Original MAE {} Corrected MAE {}'.format(str(mae[0]), str(mae_corr[0])))

input_1 = np.loadtxt('../ornekler/pair3_mae2.80/67_rhy1_per1964742_fail_grade1_onsets.txt')
input_2 = np.loadtxt('../ornekler/pair3_mae2.80/67_rhy1_ref2913742_grade4_onsets.txt')

input_1 = input_1[np.newaxis,:]
input_2 = input_2[np.newaxis,:]

pred = model.predict([input_1, input_2])
mae = abs(pred-1)

input_1_corr = np.loadtxt('../ornekler/pair3_mae2.80/67_rhy1_per1964742_fail_grade1_onsets_corrected.txt')
input_2_corr = np.loadtxt('../ornekler/pair3_mae2.80/67_rhy1_ref2913742_grade4_onsets_corrected.txt')

input_1_corr = input_1_corr[np.newaxis,:]
input_2_corr = input_2_corr[np.newaxis,:]

pred_corr = model.predict([input_1_corr, input_2_corr])
mae_corr = abs(pred_corr-1)

print('Original MAE {} Corrected MAE {}'.format(str(mae[0]), str(mae_corr[0])))


