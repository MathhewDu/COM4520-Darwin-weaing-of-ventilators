
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from scipy import interpolate
import re
os.getcwd()
def interpolate_windows(windows_df,windows_size=360):
    windows_num=round(windows_df.shape[0]/windows_size)
    feature_num=windows_df.shape[1]
    feature_windows_list=[[]for i in range(feature_num)]
    for window_idx in range(windows_num):
        window_df = windows_df.iloc[window_idx*windows_size:(window_idx+1)*windows_size,:]
        for feature_idx in range(feature_num):
            feature_window=np.array(window_df.iloc[:,feature_idx])
            not_nan = np.where(np.isnan(feature_window)==False)[0]
            if len(not_nan)==0:
                feature_windows_list[feature_idx].append(np.array([0.0]*windows_size))
            elif len(not_nan)==1:
                not_nan_value = feature_window[not_nan][0]
                feature_windows_list[feature_idx].append(np.array([not_nan_value]*windows_size))
            elif len(not_nan)==windows_size:
                feature_windows_list[feature_idx].append(feature_window)
            else:
                if not_nan[0]>0:
                    recursive_left = list(range(0,not_nan[0]))
                    feature_window[recursive_left] = feature_window[not_nan[0]]
                    not_nan = np.concatenate((recursive_left,not_nan),axis=0)
                if not_nan[-1]<windows_size:
                    recursive_right = list(range(not_nan[-1],windows_size))
                    feature_window[recursive_right] = feature_window[not_nan[-1]]
                    not_nan = np.concatenate((not_nan,recursive_right),axis=0)
                not_nan_value = feature_window[not_nan]
                interp1d=interpolate.interp1d(not_nan,not_nan_value,kind='nearest')
                feature_windows_list[feature_idx].append(interp1d(list(range(0,windows_size))))
    for feature_idx in range(feature_num):            
        feature_windows_list[feature_idx]=np.concatenate(feature_windows_list[feature_idx])
    interpolate_dataset=np.array(feature_windows_list)
    return interpolate_dataset

def normalization(df):
    float_column = ['PS_Above_PEEP', 'EPAP_PEEP', 'SET_FIO2',
    'Spont_RR', 'Deadspace', 'SBI', 'P01', 'HeartRate', 'ArtSystollic',
    'ArtMAP']
    range_list=[]
    min_list=[]
    for column in float_column:
        label_column = df[column]
        min_list.append(np.nanmin(label_column))
        _range = np.nanmax(label_column) - np.nanmin(label_column)
        range_list.append(_range)
        normalized_data = (label_column - np.nanmin(label_column)) / _range
        df[column] = normalized_data
    return df,min_list,range_list

def inverse_normalization(min_list,range_list,predict_data):
    predict_data_inverse=[]
    for i in range(10):
        predict_data_inverse.append(predict_data[i]*range_list[i]+min_list[i])
    return predict_data_inverse

def load(file_name):
    token_pattern=r'[0-9]'
    pattern=re.compile(token_pattern)
    windows_size=int(''.join(pattern.findall(file_name)))
    windows_df=pd.read_csv(file_name)
    remove_column=['Time', 'ID', 'AdmissionDate', 'DischargeDate', 
        'VentMode', 'timestamps', 'PH', 'PaCO2', 'PaO2', 'LactateABG', 'Temperature','empty_col','HospOutcome','CCOutcome']
    for column in remove_column:
        windows_df.pop(column)
    windows_df.replace({' A':1.0,' D':0.0},inplace=True)
    interpolate_df=pd.DataFrame(interpolate_windows(windows_df,windows_size=windows_size).T,columns=['PS_Above_PEEP', 'EPAP_PEEP', 'SET_FIO2',
        'Spont_RR', 'Deadspace', 'SBI', 'P01', 'HeartRate', 'ArtSystollic','ArtMAP'])
    return interpolate_df,windows_size
  
def load_with_outcome(file_name):
    token_pattern=r'[0-9]'
    pattern=re.compile(token_pattern)
    windows_size=int(''.join(pattern.findall(file_name)))
    windows_df=pd.read_csv(file_name)
    remove_column=['Time', 'ID', 'AdmissionDate', 'DischargeDate', 
        'VentMode', 'timestamps', 'PH', 'PaCO2', 'PaO2', 'LactateABG', 'Temperature','empty_col','HospOutcome']
    for column in remove_column:
        windows_df.pop(column)
    windows_df.replace({' A':1.0,' D':0.0},inplace=True)
    interpolate_df=pd.DataFrame(interpolate_windows(windows_df,windows_size=windows_size).T,columns=['CCOutcome','PS_Above_PEEP', 'EPAP_PEEP', 'SET_FIO2',
        'Spont_RR', 'Deadspace', 'SBI', 'P01', 'HeartRate', 'ArtSystollic','ArtMAP'])
    return interpolate_df,windows_size

class MyDataset(Dataset):
  def __init__(self, file_name, train_rate = 0.8, be_normalize=True):
    load_df,windows_size =load(file_name)
    x_train_set=[]
    if train_rate>0:
      train_windows_num=round(load_df.shape[0]*train_rate//windows_size)
      train_df=load_df.iloc[:train_windows_num*windows_size,:]
      if be_normalize:
        train_df,train_min,train_range=normalization(train_df)
      for windows_idx in range(train_windows_num):
        x_train_set.append(train_df.iloc[windows_idx*windows_size:(windows_idx+1)*windows_size,:].values)
    self.x_train=torch.tensor(np.array(x_train_set),dtype=torch.float32).cuda()
    self.train_range=train_range
    self.train_min=train_min
    x_test_set=[]
    if train_rate<1:
      test_df=load_df.iloc[train_windows_num*windows_size:,:]
      if be_normalize:
        test_df,test_min,test_range=normalization(test_df)
      test_windows_num=round(load_df.shape[0]/windows_size)-train_windows_num
      for windows_idx in range(test_windows_num):
        x_test_set.append(test_df.iloc[windows_idx*windows_size:(windows_idx+1)*windows_size,:].values)
    self.x_test=torch.tensor(np.array(x_test_set),dtype=torch.float32).cuda()
    self.test_range=test_range
    self.test_min=test_min
      
  def __len__(self,testset=False):
    if testset:
        return (self.x_test.shape[0])
    return (self.x_train.shape[0])

  def __getitem__(self,idx,testset=False):
    if testset:
      return (self.x_test[idx])
    return (self.x_train[idx])


class ClassificationDataset(Dataset):
  def __init__(self, file_name, train_rate = 0.8, be_normalize=True, predict_period=30):
    load_df,windows_size = load_with_outcome(file_name)
    windows_size=predict_period
    x_train_set=[]
    if train_rate>0:
      train_windows_num=round(load_df.shape[0]*train_rate//windows_size)
      train_df=load_df.iloc[:train_windows_num*windows_size,:]
      if be_normalize:
        train_df,train_min,train_range=normalization(train_df)
      for windows_idx in range(train_windows_num):
        x_train_set.append(train_df.iloc[windows_idx*windows_size:(windows_idx+1)*windows_size,:].values)
    self.x_train=torch.tensor(np.array(x_train_set),dtype=torch.float32).cuda()
    self.train_range=train_range
    self.train_min=train_min
        
    x_test_set=[]
    if train_rate<1:
      test_df=load_df.iloc[train_windows_num*windows_size:,:]
      if be_normalize:
        test_df,test_min,test_range=normalization(test_df)
      test_windows_num=round(load_df.shape[0]/windows_size)-train_windows_num
      for windows_idx in range(test_windows_num):
        x_test_set.append(test_df.iloc[windows_idx*windows_size:(windows_idx+1)*windows_size,:].values)
    self.x_test=torch.tensor(np.array(x_test_set),dtype=torch.float32).cuda()
    self.test_range=test_range
    self.test_min=test_min
    
  def __len__(self,testset=False):
    if testset:
        return (self.x_test.shape[0])
    return (self.x_train.shape[0])

  def __getitem__(self,idx,testset=False):
    if testset:
      return (self.x_test[idx])
    return (self.x_train[idx])


