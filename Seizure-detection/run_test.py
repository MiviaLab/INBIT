'''
paper version means that we are considering seizure records only with LOOCV
'''

import pdb # per breakpoint()
import mne # per EEG
import random
from sklearn.metrics import balanced_accuracy_score
import os, glob # per gestione path
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
from torchsummary import summary
import matplotlib.pyplot as plt #per immagini
from typing import List, Union, Tuple, Any
from torch.utils.data.dataset import ConcatDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
from graphviz import Digraph
from sklearn.metrics import confusion_matrix
from scipy.signal import butter, filtfilt, welch
from sklearn import preprocessing




# Model
class LightCnn(nn.Module):
    def __init__(self, output_size = 2):
        super().__init__() 
        
        self.conv_layer1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = (1, 4), padding = 'same')    
        self.bn1 = nn.BatchNorm2d(4)
        self.conv_layer2 = nn.Conv2d(in_channels = 4, out_channels = 16, kernel_size = (1, 16), padding = 'same')  
        self.bn2 = nn.BatchNorm2d(16)
        self.conv_layer3 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (1, 8), padding = 'same')  
        self.bn3 = nn.BatchNorm2d(16)
        self.conv_layer4 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (16, 1), padding = 'same')
        self.bn4 = nn.BatchNorm2d(16)
        self.conv_layer5 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (8, 1), padding = 'same') 
        self.bn5 = nn.BatchNorm2d(16)
        self.linear = nn.Linear(in_features = 16, out_features = output_size) 

       
        self.maxpool1 = nn.MaxPool2d(kernel_size = (1, 8))
        self.maxpool2 = nn.MaxPool2d(kernel_size = (1, 4))
        self.maxpool3 = nn.MaxPool2d(kernel_size = (1, 4))
        self.maxpool4 = nn.MaxPool2d(kernel_size = (2, 1))
        self.AdAvPool = nn.AdaptiveAvgPool2d(output_size = 1)

        self.ReLU = nn.ReLU()
        self.flatten = nn.Flatten()               
        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, x, verbose = False):
     
        x = x.unsqueeze(-3)

        x = self.ReLU(self.conv_layer1(x))
        x = self.maxpool1(self.bn1(x))
        x = self.ReLU(self.conv_layer2(x))
        x = self.maxpool2(self.bn2(x))
        x = self.ReLU(self.conv_layer3(x))
        x = self.maxpool3(self.bn3(x))
        x = self.ReLU(self.conv_layer4(x))
        x = self.maxpool4(self.bn4(x))
        x = self.ReLU(self.conv_layer5(x))
        x = self.AdAvPool(self.bn5(x))       
        out = self.linear(self.flatten(x))
        # probs = self.softmax(out)
        return out

####################################################################################
####################################################################################
                                    # Code
####################################################################################
####################################################################################

# check if a GPU is available
with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('I am using %s now.' %device)

# Global information
path = '/user/rferrara/Project/test_channels/LOrecordOut/2ch_Github'
fsample = 256
n_channels = 2
length_windows = 4 
window_size = length_windows*fsample
overlap_test = 0.5 # two-second overlap
step_size = int(window_size * (1 - overlap_test))
buffer_length = 3 # buffer of three 338 consecutive four-second sliding windows

df = pd.DataFrame(columns=['path', 'specificity', 'sensitivity', 'accuracy', 'balanced accuracy', 'total_delays', 'num_seizures'])

for SUBJECT_ID in os.listdir(os.path.join(path, 'models')):
    if SUBJECT_ID == 'chb01':
        CH1 = 7 # 'P3-O1' 
        CH2 = 13 # 'F8-T8'
    elif SUBJECT_ID == 'chb02':
        CH1 = 2 # 'T7-P7'
        CH2 = 14 # T8-P8
    elif SUBJECT_ID == 'chb03':
        CH1 = 2 # 'T7-P7'
        CH2 = 15 # 'P8-O2'
    elif SUBJECT_ID == 'chb04':
        CH1 = 17 ## CZ-PZ; 
        CH2 = 14 # 'T8-P8'
    elif SUBJECT_ID == 'chb05':
        CH1 = 17 # CZ-PZ
        CH2 = 15 # 'P8-O2'
    elif SUBJECT_ID == 'chb06': 
        CH1 = 13 # 'F8-T8'
        CH2 = 14 # 'T8-P8' 
    elif SUBJECT_ID == 'chb07':
        CH1 =  17 # CZ-PZ
        CH2 =  15 # 'P8-O2'
    elif SUBJECT_ID == 'chb08':
        CH1 = 9 # 'F4-C4'
        CH2 = 15 # 'P8-O2'
    elif SUBJECT_ID == 'chb09': 
        CH1 = 3 # 'P7-O1'
        CH2 = 14 # 'T8-P8' 
    elif SUBJECT_ID == 'chb10':
        CH1 = 2 # 'T7-P7'
        CH2 = 14 # 'T8-P8' 
    elif SUBJECT_ID == 'chb11': 
        CH1 = 2  # 'T7-P7'
        CH2 = 16 # 'FZ-CZ' 
    elif SUBJECT_ID == 'chb12':
        CH1 = 5 # 'F3-C3'
        CH2 = 9 # 'F4-C4' 
    elif SUBJECT_ID == 'chb13': 
        CH1 = 0 # 'FP1-F7'
        CH2 = 11 # 'P4-O2'
    elif SUBJECT_ID == 'chb14':
        CH1 = 0 # 'FP1-F7'
        CH2 = 14 # 'T8-P8' 
    elif SUBJECT_ID == 'chb15': 
        CH1 = 17 # CZ-PZ
        CH2 = 2 # 'T7-P7'
    elif SUBJECT_ID == 'chb16':
        CH1 = 3 # 'P7-O1'
        CH2 = 5 # 'F3-C3' 
    elif SUBJECT_ID == 'chb17':
        CH1 = 10 # 'C4-P4'
        CH2 = 11 # 'P4-O2'
    elif SUBJECT_ID == 'chb18': 
        CH1 = 10 # 'C4-P4'
        CH2 = 14 # 'T8-P8' 
    elif SUBJECT_ID == 'chb19': 
        CH1 = 17 # CZ-PZ; 
        CH2 = 20 #'FT10-T8'
    elif SUBJECT_ID == 'chb20':
        CH1 = 15 # 'P8-O2'
        CH2 = 18 # 'T7-FT9'
    elif SUBJECT_ID == 'chb21':
        CH1 = 6 # 'C3-P3'
        CH2 = 14 # 'T8-P8' 
    elif SUBJECT_ID == 'chb22':
        CH1 = 2 # 'T7-P7'
        CH2 = 9 # 'F4-C4'
    elif SUBJECT_ID == 'chb23':
        CH1 = 2 # 'T7-P7'
        CH2 = 15 # 'P8-O2'
    elif SUBJECT_ID == 'chb24':
        CH1 = 10 # 'C4-P4' 
        CH2 = 13 # 'F8-T8'

    else: continue

    subject_id = SUBJECT_ID
    path_test_records = [elemento for elemento in os.listdir(os.path.join(path, 'test_folds', subject_id)) if "test" in elemento]
    path_test_labels = [elemento for elemento in os.listdir(os.path.join(path, 'test_folds', subject_id)) if "label" in elemento]
    path_saved_models = [elemento for elemento in os.listdir(os.path.join(path, 'models', subject_id)) if "model.pth" in elemento]
  
    for folds in range(len(path_test_records)):
        model = LightCnn()
        test_path = os.path.join(path, 'test_folds', subject_id, path_test_records[folds])
        labels_path = os.path.join(path, 'test_folds', subject_id, path_test_labels[folds])

        selected_fold = test_path.split('_')[-2] # Verify the fold number
        model.load_state_dict(torch.load(os.path.join(path, 'models', subject_id, [file for file in path_saved_models if f'fold_{selected_fold}' in file][0])))
        
        print('\n test: ', test_path)
        print('fold: ', [file for file in path_saved_models if f'fold_{selected_fold}' in file][0])
        test_fold = np.load(test_path)
        test_labels = torch.tensor(np.load(labels_path))
        windows_test = torch.stack([torch.tensor(test_fold[:, i:i + window_size]) for i in range(0, test_fold.shape[1] - window_size + 1, step_size)])

        model.eval()
        
        with torch.no_grad():
            outputs = []
            labels = []
            acc = []
            acc_bal = []
            total_delays=[]

            signal_csv = pd.read_csv(os.path.join(path, 'GT', subject_id  +".csv"), sep=';')
            idx = signal_csv[signal_csv['Name of file'] ==  path_test_records[folds].split('_fold_')[0].replace(".npy", ".edf")].index[0]
            total_signal = test_fold
            time = np.arange(0, len(total_signal[0, :]), 1)
            time_interval= torch.stack([torch.tensor([time[i:i + window_size][0], time[i:i + window_size][-1]]) for i in range(0, time.shape[0] - window_size + 1, step_size)])
            num_seizure = signal_csv['Numb of seizures'][idx]
            
            if num_seizure > 1:
                starts = signal_csv['Start (sec)'][idx].split("-") 
                ends = signal_csv['End (sec)'][idx].split("-")
                for rep in range(num_seizure):
                    start_idx = int(starts[rep]) *fsample
                    end_idx = int(ends[rep])*fsample
                    print("Period signal: ", [start_idx, end_idx], " samples :",[int(start_idx)/fsample, int(end_idx)/fsample], "sec" )       
            else:
                start_idx = int(signal_csv['Start (sec)'][idx]) *fsample
                end_idx = int(signal_csv['End (sec)'][idx])*fsample
                if end_idx == 0: end_idx =len(total_signal[0, :])
                print("Period signal: ", [start_idx, end_idx], "-",[int(start_idx)/fsample, int(end_idx)/fsample] )
            
           
            win_vector = []
            label_vector =[]

            output_window = [] 
            label_window = []

            flag_output = 0   # Flag variable used to identify the output of the segment (buffer of 3 consecutive windows). Es: outputs = [0, 0, 0] -> flag_output = 0
            flag_label = 0    # Flag variable used to identify the label of the segment (buffer of 3 consecutive windows). Es: labels = [0, 0, 0] -> flag_label = 0
            previous_flag = 0 # Flag variable used to identify the class of the previous segment (buffer of 3 consecutive windows)
            print(len(windows_test), '-', len(test_labels))
            if len(windows_test) < len(test_labels): data = windows_test
            else:  data = test_labels
            for k in range(len(data)):
                X = windows_test[k]
                y = test_labels[k]
                t = time_interval[k]
                
                X = X.unsqueeze(0).float()
                y = y.long()
            
                o = model(X)
                output_window.append(torch.argmax(o).item())
                label_window.append(y.item())

                win_vector.append(torch.argmax(o).item())
                label_vector.append(y.item())
                
                
                if len(win_vector) > buffer_length:
                    # ground truth over three windows

                    # Definition of the label based on three consecutive windows
                    # scenario involving three consecutive class 0 windows: inter-ictal phase
                    if all(x == 0  for x in label_vector): 
                        label = 0
                        flag_label = 0

                    # scenario involving three consecutive class 1 windows: ictal phase
                    elif all(x == 1 for x in label_vector): 
                        if flag_label ==0:
                            # If the flag_label was 0, the beginning of the ictal phase is marked
                            GT_start_time = t[0]
                        label = 1
                        flag_label = 1

                    # scenario in which no three consecutive windows belong to the same class
                    else: 
                        if flag_label == 0: label = 0 # in case of class 0, the label assigned is 0
                        if flag_label == 1: label = 1 # in case of class 1, the label assigned is 1
                    
                    # Note that the label changes only when there are three consecutive windows belonging to the same class.


                    # Definition of the model output based on three consecutive windows
                    if all(x == 0  for x in win_vector):
                        # if previous_flag == 1: # se stavo in fase seizure identifico la fine dell'attacco epilettico
                        #     print(f'predicted end seizure: {t[0]} - {t[0]/fsample} ')
                        flag_output = 0
                    elif all(x == 1  for x in win_vector):
                        if previous_flag == 0: # If the previous segment belonged to class 0, the beginning of the epileptic seizure is identified.
                            print(f'predicted start seizure: {int(t[0])} - {int(t[0])/fsample} ')
                            if num_seizure > 1:
                                delay = []
                                for rep in range(num_seizure):
                                    start_idx = int(starts[rep]) *fsample
                                    end_idx = int(ends[rep])*fsample
                                    delay.append(np.abs(int(start_idx)/fsample - int(t[0])/fsample))
                                print("Delay: ", min(delay))
                                total_delays.append(min(delay))
                                
                            else:
                                print("Delay: ", np.abs(int(start_idx)/fsample - int(t[0])/fsample))
                                total_delays.append(np.abs(int(start_idx)/fsample - int(t[0])/fsample)) 
                                # compute and save the delay as the time difference between the ground truth (int(start_idx)/fsample)
                                # and the detected onset of the ictal event at time t[0] (int(t[0])/fsample).
                        
                        flag_output = 1
                    
                    outputs.append(flag_output) # save the outputs of the segments
                    labels.append(label)        # save the labels of the segments
                    acc.append((flag_output == label)) # compute and save the accuracy of the segments

                    previous_flag = flag_output # updated 

                    # remove the first of the three windows in the buffer to evaluate the subsequent one
                    win_vector.pop(0) 
                    label_vector.pop(0)
                    # print("it-k: ", k,  "time: ", t, "label: ", label, "output:", flag_output) 

                    # if not num_seizure == 0: # if there is at least one epileptic seizure
                    #     if end_idx:
                    #         if t[0] > start_idx and t[0] < ( end_idx + 60*fsample): 
                    #             print("it-k: ", k,  "time: ", t, "label: ", label, "output:", flag_output) 
                   
                    #         #if t[0] > end_idx and t[0] < ( end_idx + 60*fsample): 
                                
                    #             #  breakpoint() 
                    #             #labels2.pop()
                    #             #outputs2.pop()
                    #             #acc2.pop()
                    #             print('\n campione eliminato -', t[0])

                    #     elif ends:
                    #         for rep in range(ends):
                    #             end_idx = int(ends[rep])*fsample  
                    #             if t[0] > start_idx and t[0] < ( end_idx + 60*fsample): 
                    #                 print("it-k: ", k,  "time: ", t, "label: ", label, "output:", flag_output) 
                                    
                    #                 #if t[0] > end_idx or t[0] < ( end_idx + 60*fsample): 
                    #                     # labels2.pop()
                    #                     # outputs2.pop()
                    #                     # acc2.pop()
                    #                     # print('\n campione eliminato -', t[0])
                    
                    #if not flag_output == label: print("it-k2: ", k,  "time: ", t, "label: ", label, "output:", flag_output) 
        

            acc_bal.append(balanced_accuracy_score(labels, outputs)) 
            accuracy = np.array(acc, dtype=int).mean()

            cm = confusion_matrix(labels, outputs)
            print("\n Confusion matrix: ")
            print(cm)

            if cm.ravel().shape !=(1,):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)

            else: 
                tn = cm.ravel()
                fp = 0
                fn = 0
                tp = 0
                sensitivity = 0
                specificity = tn / (tn + fp)
                acc_bal = 0 # because all correctly classified samples belong to class 0

            
            print(f'\n Balanced test set: specificity: {specificity}, sensitivity: {sensitivity}, accuracy: { accuracy}, balanced accuracy: {acc_bal}, total_delays: {total_delays}')
            
            if total_delays == []: total_delays = ''
            
            new_line = pd.DataFrame({'path': str(path_test_records[folds].split('_fold_')[0]), 'specificity': specificity, 'sensitivity': sensitivity, 'accuracy': accuracy, 'balanced accuracy': acc_bal, 'total_delays': str(total_delays), 'num_seizures': num_seizure})
 
            df = pd.concat([df, new_line], ignore_index=True)

            df.to_csv('Results.csv', index=False)
