

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 20:55:20 2022

@author: basakesin
"""


#Install requirements
#Load required packages
import matplotlib.pyplot as plt
import numpy as np
from numpy import sort
import os
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor, AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error,  mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import shutil, sys
import xgboost as xgb
from zipfile import ZipFile
from joblib import Parallel, delayed
import multiprocessing
from pathlib import Path


def voting_regressor(mae, test_predictions,column_names):
    sorted_mae_index = np.argsort(mae)
    number_of_interested_cols = 3
    interested_cols = []
    for i in range(0,number_of_interested_cols):
        interested_cols.append(column_names[sorted_mae_index[i]])
    interested_test_predictions = test_predictions[interested_cols]
    test_predictions['VR'] = interested_test_predictions.sum(axis=1) / interested_test_predictions.shape[1]
    return test_predictions

def regression_test(train_data_path, test_data_path, save_results_to):
    

    #Load train and test data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)


    y_train = np.array(train_data.grade)
    y_test = np.array(test_data.grade)
    #Normalize the data
    scaler = MinMaxScaler()
    #Learn scaler from train data and apply it to train and test data
    X_train = scaler.fit_transform(train_data.iloc[:,2:-1])
    X_test = scaler.transform(test_data.iloc[:,2:-1])
    
    #Design Regression Methods
    random_state = 7
    r1 = LinearRegression()
    r2 = RandomForestRegressor(random_state=random_state)
    r3 = AdaBoostRegressor(random_state=random_state)
    r4 = BaggingRegressor(random_state=random_state)
    r5 = ExtraTreesRegressor(random_state=random_state)
    r6 = xgb.XGBRFRegressor(random_state=random_state)

    models = []
    models.append(('LR', LinearRegression()))
    models.append(('RF',RandomForestRegressor(random_state=random_state)))
    models.append(('Adaboost', AdaBoostRegressor(random_state=random_state)))
    models.append(('Bagging', BaggingRegressor(random_state=random_state)))
    models.append(('ETR',ExtraTreesRegressor(random_state=random_state)))
    models.append(('XGBoost', xgb.XGBRFRegressor(random_state=random_state)))

    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    column_names = ['LR', 'RF', 'Adaboost','Bagging', 'ETR','XGBoost', 'VR','DE1','DE2']
    results_on_test = pd.DataFrame(np.nan, index = ['MAE','MSE','R2'], columns = column_names[0:7])
    
    cross_val_accuracy_mae = pd.DataFrame(index = range(0,5),columns=column_names[:-2])
    cross_val_accuracy_mse = pd.DataFrame(index = range(0,5),columns=column_names[:-2])
    cross_val_accuracy_r2 = pd.DataFrame(index = range(0,5),columns=column_names[:-2])
    mae = []
    mse = []
    r2_s = []
    MAE_bar = pd.DataFrame(columns = column_names)
    test_predictions = pd.DataFrame(columns=column_names)
    test_predictions['Ref_file'] = test_data['Ref_file']
    test_predictions['Per_file'] = test_data['Per_file']
    test_predictions['Grade'] = y_test
    cols = test_predictions.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    test_predictions = test_predictions[cols]
    #Apply all regression methods and calculate performance metrics
    
    for name,model in models:
        print(name)
        model.fit(X_train,y_train)
        cross_val_accuracy_mae[name] = abs(cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv))
        cross_val_accuracy_mse[name] = abs(cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv))
        cross_val_accuracy_r2[name] = cross_val_score(model, X_train, y_train, scoring='r2', cv=cv)
        y_pred = model.predict(X_test)
        test_predictions[name]=y_pred
        mae.append(mean_absolute_error(y_test,y_pred))
        mse.append(mean_squared_error(y_test,y_pred))
        MAE_bar[name] = abs(y_test-y_pred)
        r2_s.append(r2_score(y_test,y_pred))
    

    test_predictions = voting_regressor(mae, test_predictions,column_names) 
    mae.append(mean_absolute_error(y_test,test_predictions['VR']))
    mse.append(mean_squared_error(y_test,test_predictions['VR']))
    MAE_bar[name] = abs(y_test-test_predictions['VR'])
    r2_s.append(r2_score(y_test,test_predictions['VR']))
    test_predictions.to_csv(os.path.join(save_results_to,'test_predictions.csv'))
    

    #Write performance metrics to dataframe
    results_on_test.loc['MAE']=mae
    results_on_test.loc['MSE']=mse
    results_on_test.loc['R2']=r2_s
    MAE_bar['DE1'] = abs(y_test- np.full_like(y_test, 2))
    MAE_bar['DE2'] = abs(y_test- np.full_like(y_test, 3))
    means = MAE_bar.mean()
    stds = MAE_bar.std()
    mins = MAE_bar.min()
    maxes = MAE_bar.max()
    #Plot error bars
    plt.errorbar(column_names, means, stds, fmt='ok', lw=3)
    plt.errorbar(column_names, means, [means - mins, maxes - means],
                 fmt='.k', ecolor='gray', lw=1)
    plt.xticks(rotation=20)
    plt.ylabel("Average MAE")
    plt.savefig(os.path.join(save_results_to,'Fig1.eps'), format='eps', dpi=1200)
    #files.download("Fig3.eps") 
    #plt.title('Algorithms performances over test set')
    plt.close('all')
    #Write all results on a text file
    results_file = os.path.join(save_results_to, 'results.txt')
    tem = sys.stdout
    sys.stdout = f = open(results_file, 'a')
    
    
    print('Cross-validation MAE on train data')
    print(cross_val_accuracy_mae)
    
    print('Cross-validation MSE on train data')
    print(cross_val_accuracy_mse)
    
    print('Cross-validation R2 score on train data')
    print(cross_val_accuracy_r2)
    
    print('Results on Test Set')
    print(results_on_test)
    
    
    print('Average MAE of cross validation')
    print(cross_val_accuracy_mae.mean())
    
    print('Standart Deviation of Each Algoritm on Cross-Validation')
    print(cross_val_accuracy_mae.std())    
    
    
    print('Average MSE of cross validation')
    print(cross_val_accuracy_mse.mean())
    
    print('Standart Deviation of Each Algoritm on Cross-Validation')
    print(cross_val_accuracy_mse.std())   
    
    
    print('Average R2 Score of cross validation')
    print(cross_val_accuracy_r2.mean())
    
    print('Standart Deviation of Each Algoritm on Cross-Validation')
    print(cross_val_accuracy_r2.std())    
    sys.stdout = tem
    f.close()
    

    #Calculate the feature importances according to the Random Forest Regressor
    chosed_model = r2
    feature_names = train_data.columns
    feature_names = feature_names[2::]
    chosed_model.fit(X_train,y_train)
    
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 2), chosed_model.feature_importances_), feature_names), 
                 reverse=True))
    
    sorted_idx = chosed_model.feature_importances_.argsort()
    
    y_ticks = np.arange(0, len(feature_names)-1)
    
    sorted_feature_names = np.empty_like(feature_names)
    t = 0
    for i in sorted_idx:
      sorted_feature_names[t] = feature_names[i]
      t = t+1

    
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 2), chosed_model.feature_importances_), feature_names), 
                 reverse=True))

    
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(y_ticks, chosed_model.feature_importances_[sorted_idx])
    ax.set_xticklabels(sorted_feature_names)
    ax.set_xticks(y_ticks)
    plt.rc('font', size=20) 
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    
   
    plt.savefig(os.path.join(save_results_to,'Fig2.eps'), format='eps', dpi=1200)
    fig.tight_layout()
    plt.show()
     
    plt.close('all')
    thresholds = sort(chosed_model.feature_importances_)
    k = X_train.shape[1]
    column_names = ['LR', 'RF', 'Adaboost','Bagging', 'ETR','XGBoost','VR']
    thresholds_df_mae = pd.DataFrame(np.nan, index=range(0,k), columns=column_names)
    thresholds_df_mse = pd.DataFrame(np.nan, index=range(0,k), columns=column_names)
    thresholds_df_r2 = pd.DataFrame(np.nan, index=range(0,k), columns=column_names)
    
    for thresh in thresholds:
        k = k-1
        selection = SelectFromModel(chosed_model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        select_X_test = selection.transform(X_test)
        # train model
        
        results_on_test_feature_imp = pd.DataFrame(np.nan,index = [0],columns = column_names)
        results_on_test_feature_imp_MSE = pd.DataFrame(np.nan,index = [0],columns = column_names)
        results_on_test_feature_imp_R2 = pd.DataFrame(np.nan,index = [0],columns = column_names)
        for name,model in models:
            model.fit(select_X_train,y_train)
            y_pred = model.predict(select_X_test)
            results_on_test_feature_imp[name] = mean_absolute_error(y_test,y_pred)
            results_on_test_feature_imp_MSE[name] = mean_squared_error(y_test,y_pred)
            results_on_test_feature_imp_R2[name] = r2_score(y_test,y_pred)
            thresholds_df_mae.loc[k] = np.array(results_on_test_feature_imp)
            thresholds_df_mse.loc[k] = np.array(results_on_test_feature_imp_MSE)
            thresholds_df_r2.loc[k] = np.array(results_on_test_feature_imp_R2)
    
    plt.figure(figsize=(20,10))
    plt.rc('font',size=12)
    
    plt.plot(range(1,X_train.shape[1]+1),thresholds_df_mae.values)
    plt.legend(column_names)
    plt.xlabel('Number of Features')
    plt.ylabel('MAE')
    plt.xticks(range(1,X_train.shape[1]+1))
    plt.savefig(os.path.join(save_results_to,'Fig3.eps'), format='eps', dpi=1200)
    plt.close('all')
    

def extract_zip(zipfile_path, target_folder_path):
    zip = ZipFile(zipfile_path, 'r')
    zip.extractall(target_folder_path)
    
def baseline_regression(ind,i,target_folder_path,author_name):
   zipfile_path = os.path.join(target_folder_path , i)
   temp = i.split(".")
   extraction_path = os.path.join(target_folder_path , temp[0])
   try:
       os.makedirs(extraction_path)
   except FileExistsError:
       print(extraction_path + 'directory exists, I am overwriting results on it')
   extract_zip(zipfile_path, extraction_path)
   train_data_path = os.path.join(extraction_path, 'trainData.csv')
   test_data_path = os.path.join(extraction_path, 'testData.csv')
   if ind<3:
       file_name = 'annotations_'+author_name[ind]
   else:
       file_name = 'annotations_'+author_name[ind%len(author_name)]+'_withCorrectedOnsets'
   save_results_to = os.path.join(Path(os.getcwd()).parent,'results/rhythm/baseline_regression/', file_name)
   try:
       os.makedirs(save_results_to)
   except FileExistsError:
       print(save_results_to + 'directory exists, I am overwriting results on it')
   regression_test(train_data_path, test_data_path, save_results_to)
   shutil.rmtree(os.path.join(extraction_path))
    
    
def main():
  
    data_annotations_files = ['rhythm_data4ML_0.zip','rhythm_data4ML_1.zip','rhythm_data4ML_2.zip','rhythm_data4ML_0_withCorrectedOnsets.zip','rhythm_data4ML_1_withCorrectedOnsets.zip','rhythm_data4ML_2_withCorrectedOnsets.zip']
    author_name = ['0','1','2']
    target_folder_path = os.path.join(Path(os.getcwd()).parent,'data/rhythm/')      
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(baseline_regression)(ind,i,target_folder_path,author_name) for ind, i in enumerate(data_annotations_files))

if __name__ == '__main__':
   main()
   
   
