import pdb # per breakpoint()
import mne # per EEG
import os, glob # per gestione path
from pathlib import Path
import matplotlib.pyplot as plt #per immagini
import numpy as np
import pandas as pd
from pandas import DataFrame

dataset_path = "/user/rferrara/Project/physionet.org/files/chbmit/1.0.0"
# Note that when downloading the CHB dataset, the data is provided in .edf format in a folder <personal_path>/1.0.0. 
# To run this code, all files in the format record_name.edf.seizures have been renamed to record_name.edf

def features_extraction(subject_path, features_path):
    #df = pd.DataFrame(columns = ['path', 'length'])
    for subject_file in os.listdir(subject_path): # es. subject_file = 'chb01_01.edf'
        print("subject file: ", subject_file)
       
        [data, time] = check_channels(record = mne.io.read_raw_edf(os.path.join(subject_path, subject_file))) 
       
        if os.path.exists(os.path.join(features_path, subject_file.replace(".edf", ".npy"))):
            continue
      
        np.save(os.path.join(features_path, subject_file.replace(".edf", "")), data)
   

def monopolar2bipolar(record, channels): # compute the electrode pair potential differences
    pair_ch = channels[0].split("-") 
    [data, time] = record[:, :] 
    new_data = np.zeros((len(channels), data.shape[1]))
    idx = 0
    for ch in range(len(channels)):
        flag_1 = False
        flag_2 = False
        pair_ch = channels[ch].split("-") 
        plt.figure()
        for i in range(record.info["nchan"]):
            if pair_ch[0] in record.info["ch_names"][i]: # channel 1 is extracted
                if flag_1 : raise RuntimeError("Error")
                flag_1= True
                signal1 = data[i, :]
                
            elif pair_ch[1] in record.info["ch_names"][i]: # channel 2 is extracted
                if flag_2 : raise RuntimeError("Error")
                flag_2 = True
                signal2 = data[i, :]

            if (flag_1 and flag_2):
                new_data[ch, :] = signal1 - signal2 # bipolar record
                flag_1 = False
                flag_2 = False
        plt.plot(new_data[ch, :])
        plt.savefig('new')
        
    return [new_data, time]


def check_channels(record):
    # List of selected channels (ordered list)
    list_of_channels=['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 
        'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'T7-FT9', 'FT9-FT10', 'FT10-T8']

    # in case of monopolar channels
    if (record.info["ch_names"][0].find("-CS") != -1) or (record.info["ch_names"][0].find("-") == -1): # find() restituisce -1 quando non si trova quello che si cerca
        new_list = ['FP1', 'F7', 'T7', 'P7', 'O1',  'F3', 'C3', 'P3',  'FP2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8', 'FZ', 'CZ', 'PZ', 'T7', 'FT9', 'FT10']
        print(record.info["ch_names"])
        [raw_data, time] = record[:, :]
        data = np.zeros((len(new_list), raw_data.shape[1]))
        for ch in range(len(new_list)):
            for i in range(record.info["nchan"]):
                if new_list[ch] in record.info["ch_names"][i]:
                    if 'CP4' in record.info["ch_names"][i]: continue
                    else:
                        data[ch, :] = raw_data[i, :]
                        print(record.info["ch_names"][i])
                        print(i)
                        
        [data, time] = monopolar2bipolar(record, list_of_channels)

    else: 
        record = record.pick_channels(list_of_channels)
        for i in range(len(record.info["ch_names"])):
            # Code for handling bipolar EEG recordings with removal of duplicate channels
            if (record.info["ch_names"][i].find("-0") != -1): 
                record = record.rename_channels({record.info["ch_names"][i] : record.info["ch_names"][i].replace("-0", "")})    
        [data, time] = record[:, :]

    return [data, time]


for subject_name in os.listdir(dataset_path): 
    path_features = dataset_path.replace("1.0.0", "features")
    
    if not os.path.exists(path_features):
        os.makedirs(path_features) 

    if not os.path.exists(os.path.join(path_features, subject_name)):
        os.makedirs(os.path.join(path_features, subject_name))
    features_extraction(os.path.join(dataset_path, subject_name), os.path.join(path_features, subject_name))    