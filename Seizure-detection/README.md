# Personalized seizure detection with two-selected EEG channels 
This repository will contain the code related to the paper: Ferrara R., Giaquinto M., Percannella G., Rundo L., Saggese A. "Personalizing Seizure Detection to Individual Patients by an Optimal Selection of EEG Signals". Under review (Sensors).

This study presents a lightweight Convolutional Neural Network optimized for seizure detection in a personalized approach, using data from two channels selected based on a data-driven analysis. This analysis identifies one channel during the ictal phase and another during the inter-ictal phase.


## Run Experiments

python run_test.py 


Folder Structures

1. utils/: contains dataset processing scripts for extracting data using the MNE package and splitting it into 1-hour records.

2. models/: contains the trained models for each patient. Note that we applied a leave-one-record-out strategy (on 1-hour records), so each patient has multiple folds.

3. test_folds/: contains the test record (in .npy format) and the corresponding label vector for each fold of every patient. The test records have been normalized using the mean and standard deviation values computed from the training set.