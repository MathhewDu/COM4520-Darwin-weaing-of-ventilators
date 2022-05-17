import pandas as pd
import os
import numpy as np 
import datetime
import matplotlib.pyplot as plt
from datetime import timedelta
from random import sample
os.getcwd()

def get_unique_ids(df):
    unique_ids = df['ID'].unique()
    unique_ids.sort()
    return unique_ids

def windows_missing_value(patient_df):
    score_multiplier=5.0
    missing_num = patient_df.isna().sum()
    total_num = patient_df.shape[0]
    missing_percentage=1-missing_num/total_num
    missing_percentage=missing_percentage*score_multiplier//1
    return(missing_percentage)

def get_window(patient_df,windows_size):
    #remove all the missing line for the patient
    valid_df=patient_df[patient_df['Time']!='<NA>']
    #if no data left for the patient, then return 100% missing for the patient
    if valid_df.size==0:
        return([],0)
    date_time = pd.to_datetime(valid_df['Time'], format='%d/%m/%Y %H:%M')
    timestamps = date_time.map(datetime.datetime.timestamp)
    valid_df['timestamps']=timestamps
    valid_df=valid_df.sort_values(by=['timestamps'])
    date_time = pd.to_datetime(valid_df['Time'], format='%d/%m/%Y %H:%M')
    deltas = date_time.diff()[1:]
    windows=[0]
    i=0
    for delta in deltas:
        if delta == timedelta(minutes=1):
            windows[i]+=1
        else:
            windows.append(1)
            i+=1
    i=0
    windows_index=[]
    filter_windows=[]
    for window in windows:
        if window >= windows_size:
            filter_windows.append(window//windows_size*windows_size)
            windows_index.append(i)
        i+=1
    i=0
    df_index=[]#index from end of each window
    new_df=[]
    for windows_indexs in windows_index:
        df_index.append(sum(windows[:windows_indexs+1]))
        start=df_index[i]-filter_windows[i]+1
        end=df_index[i]+1
        new_df.append(valid_df[start:end])
        i+=1
    windows_count=int((sum(filter_windows))/windows_size)
    return(new_df,windows_count)

def getcolor(score):
    one = 255/44
    r=0.0
    g=0.0
    b=0.0
    if score<22:
        r=255
    elif score>=22 and score<34:
        r=255
        g=255-(34-score)*one*3.67
    else:
        r=255-(score-34)*one*4.4
        g=255
        
    r=r/255
    if r>1:
        r=1.0
    if r<0:
        r=0.0
        
    g=g/255
    if g>1:
        g=1.0
    if g<0:
        g=0.0
    return(r,g,b)

def generate_windows_df(df,windows_size):
    #generate new dataframe that includes all windows
    windows_df_list = []
    windows_count_per_patient={}#record the number of windows for each patient
    windows_size = windows_size#the window size is 2 hours right now, can be easily changed
    unique_ids = get_unique_ids(df)
    for id_to_keep in unique_ids:
        find_window_df = df.loc[df['ID'] == id_to_keep]
        window_for_id,windows_count = get_window(find_window_df,windows_size)
        if windows_count>0:
            windows_count_per_patient[id_to_keep]=windows_count
        windows_df_list+=window_for_id
    windows_df = pd.concat(windows_df_list, ignore_index=True)
    return(windows_df,windows_count_per_patient)


def generate_windows_mark_csv(windows_df,windows_count_per_patient,windows_size=360,drop_feature=['AdmissionDate', 'DischargeDate', 'CCOutcome', 'VentMode', 'timestamps', 'PH', 'PaCO2', 'PaO2', 'LactateABG', 'Temperature']):

    windows_names=['ID', 'Start_Time','End_Time','AdmissionDate', 'DischargeDate', 'CCOutcome', 'HospOutcome', 'VentMode',
       'PS_Above_PEEP', 'EPAP_PEEP', 'SET_FIO2', 'Spont_RR', 'Deadspace', 'SBI', 'P01'
       , 'PH', 'PaCO2', 'PaO2', 'LactateABG', 'HeartRate', 'ArtSystollic', 'ArtMAP', 
       'Temperature', 'timestamps']

    windows_mark_df=pd.DataFrame(columns=windows_names)
    windows_patient = []
    for key,value in windows_count_per_patient.items():
        windows_patient+=([key]*value)
    row_time=list(windows_df['Time'])
    time_for_window=[]
    end_for_window=[]
    for i in range(len(windows_patient)):
        index=i*windows_size
        time_for_window.append(row_time[index])
        end_for_window.append(row_time[index+windows_size-1])
    windows_mark_df['ID']=windows_patient
    windows_mark_df['Start_Time']=time_for_window
    windows_mark_df['End_Time']=end_for_window

    dflist__windows = []#it stores every 120lines in the dataframe as a windows to the list
    i=0
    for id_to_keep in windows_patient:
        dflist__windows.append(windows_df[i*windows_size:(i+1)*windows_size])
        i+=1
    id_index=0
    windows_names=['AdmissionDate', 'DischargeDate', 'CCOutcome', 'HospOutcome', 'VentMode',
        'PS_Above_PEEP', 'EPAP_PEEP', 'SET_FIO2', 'Spont_RR', 'Deadspace', 'SBI', 'P01'
        , 'PH', 'PaCO2', 'PaO2', 'LactateABG', 'HeartRate', 'ArtSystollic', 'ArtMAP', 
        'Temperature', 'timestamps']
    for df_each_windows in dflist__windows:
        for feature in windows_names:
            miss_value_percentage=windows_missing_value(df_each_windows[feature])
            windows_mark_df.loc[id_index,feature]=miss_value_percentage
        id_index+=1
    drop_feature=drop_feature
    dropped_windows_mark_df=windows_mark_df.copy()#to avoid drop on the original dataframe
    dropped_windows_mark_df=dropped_windows_mark_df.drop(columns=drop_feature)
    windows_df.to_csv('windows_df_'+str(windows_size)+'.csv',index=False,header=True)
    dropped_windows_mark_df.to_csv('windows_mark_df_'+str(windows_size)+'.csv',index=False,header=True)
    return(windows_df,windows_count_per_patient,dropped_windows_mark_df)


def get_windows_and_mark_df(df,windows_size=360):
    dropped_windows_mark_df=pd.read_csv('windows_mark_df_'+str(windows_size)+'.csv')
    windows_df,windows_count_per_patient = generate_windows_df(df,windows_size=windows_size)
    return(windows_df,windows_count_per_patient,dropped_windows_mark_df)

def get_color_windows(dropped_windows_mark_df):
    #create a list to store the color for the score
    score_color=[]
    color_windows=[]
    for index, row in dropped_windows_mark_df.iterrows():
        score_row = 0.0
        for i in range(len(row)):
            if i > 2:#calculate start from the feature 'HospOutcome'
                mark=row[i]
                if mark<=5.0:
                    if mark == 5.0:
                        mark = 4.0
                    score_row += mark
        score_color.append(score_row)
    for score in score_color:
        color_windows.append(getcolor(score))#the list store color as (0.1,0.1,0.1) which recognized by matplotlib
    return(color_windows,score_color)

#simply show some sample for the windows quality
def drawbar(start,end,color_list):
    plt.figure(figsize=(20, 2))
    plt.title('Visulization for each windows')
    plt.bar(range(end-start+1),5,color=color_list[start:end])

def drawbar_patient(dropped_windows_mark_df,windows_count_per_patient,patient,color_list):
    patient_windows=list(windows_count_per_patient.keys())
    windows_count=list(windows_count_per_patient.values())
    patient_index=patient_windows.index(patient)
    windows_start=sum(windows_count[:patient_index])
    windows_end=sum(windows_count[:patient_index+1])
    windows_start_time=list(dropped_windows_mark_df['Start_Time'])[windows_start:windows_end]
    plt.figure(figsize=(20, 2))
    #plt.figure(figsize=(2, 20))
    plt.yticks([])
    plt.xticks(rotation=90)
    plt.title('Windows Visulization for patient '+str(patient))
    plt.bar(windows_start_time,5,color=color_list[windows_start:windows_end])
    plt.show()
    


def get_windows_count(df,windows_size):
    windows_count = (df.shape[0])/windows_size
    return (int(windows_count))


def get_windows_by_patient(patient,windows_count_per_patient,windows_df,windows_size):
    patient_windows=list(windows_count_per_patient.keys())
    windows_count=list(windows_count_per_patient.values())
    patient_index=patient_windows.index(patient)
    windows_start=sum(windows_count[:patient_index])
    windows_end=sum(windows_count[:patient_index+1])
    return(windows_df[windows_start*windows_size:windows_end*windows_size])


def get_windows_by_color(score_color,windows_df,color_score_range,windows_size):
    index_list_for_score = [x for x, y in list(enumerate(score_color)) if y in color_score_range]
    windows_concat_lst=[]
    for i in index_list_for_score:
        if i < len(windows_df)-windows_size:
            windows_concat_df = windows_df[i*windows_size:i*windows_size+windows_size]
            windows_concat_lst.append(windows_concat_df)
    windows_by_color=pd.concat(windows_concat_lst)
    return(windows_by_color)


def get_windows_by_color_from_patient(patient,color_score_range,size,
                                      windows_count_per_patient,windows_df,score_color,windows_size):
    windows_from_patient = get_windows_by_patient(patient,windows_count_per_patient=windows_count_per_patient,windows_df=windows_df)
    windows_from_patient_index = list(windows_from_patient.index)
    start_index = round(windows_from_patient_index[0]/windows_size)
    end_index = round(windows_from_patient_index[-1]/windows_size)
    print(start_index,end_index)
    score_color_from_patient = score_color[start_index:(end_index+1)]
    windows_by_color_from_patient = get_windows_by_color(size=size,score_color=score_color_from_patient,
                                                         windows_df=windows_from_patient,color_score_range=color_score_range)
    return(windows_by_color_from_patient)

    


