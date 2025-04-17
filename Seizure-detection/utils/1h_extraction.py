# file che filtra i dati tra 0.5 e 45 Hz
import pdb # per breakpoint()
import mne # per EEG
import math
import os, glob # per gestione path
from pathlib import Path
import numpy as np
import pandas as pd

dataset_path = "/user/rferrara/Project/physionet.org/files/chbmit/features"
fsample = 256

for subject_id in os.listdir(dataset_path):  # es. subject_name = 'chbXX'
    print('subject_id: ', subject_id) 
  
    x = 1 # used for new 1h file naming
    new_path = dataset_path.replace("features", "1h_features")
    if not os.path.exists(new_path):
        os.makedirs(new_path) 

    if not os.path.exists(os.path.join(new_path, subject_id)): 
        os.makedirs(os.path.join(new_path, subject_id)) 

        subject_path = os.path.join(dataset_path, subject_id)
        normalization_path = os.path.join(new_path, subject_id)
        
        for subject_file in os.listdir(subject_path): # es. subject_file = 'chb01_01.edf'
            print("subject file: ", subject_file)
            signal = np.load(os.path.join(subject_path, subject_file)) 
        
            # Loop through segments of the signal, with each segment corresponding to 1 hour of data.
            for ln in range(math.ceil((signal.shape[1]/(3600*fsample)))):
                if signal.shape[1] - (ln+1)*(3600*fsample) < 0 : #  check if the remaining data is less than 1 hour (i.e., the last segment).
                    new=signal[:, (ln)*(3600*fsample):signal.shape[1]]
                    print("ln", ln, ", start: ", (ln)*(3600*fsample), ", stop: ", signal.shape[1])
                else: # otherwise, slice a full 1-hour segment.
                    new=signal[:, (ln)*(3600*fsample):((ln)*(3600*fsample) + (3600*fsample))]
                    print("ln", ln, ", start: ", (ln)*(3600*fsample), ", stop: ", (ln)*(3600*fsample) + (3600*fsample))
                print(new.shape)
                

                if x < 10: # if x is less than 10, add a leading zero (e.g., '01', '02') to maintain consistent file naming.
                    single_file = subject_id + '_0' + str(x)  
                else: # otherwise, just append the number normally (e.g., '10', '11').
                    single_file = subject_id + '_' + str(x)   
                x = x + 1  
                np.save(os.path.join(normalization_path, single_file), new) # save new 1h_features
            

         