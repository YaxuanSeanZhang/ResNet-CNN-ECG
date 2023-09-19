# Overview

 This code base was used to generate the results for CSCI 5525, group 14.

 The overarching goal was to predict conduction disorders (CD) using 12-lead ECGs.

 The final report is in the file `final_report.pdf`.

# Requirments

This project was written using Python 3.9 and requirments are listed in `requirements.txt`.

# Code Base

 ## Data Acquisition

 The primary dataset was acquired from https://physionet.org.

 If you need access to the data, it can be downloaded from here: https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip

 Alternatively, you can use the function `fetch_data` defined in `utils.py`.

 ## Data Preprocessing

 The directory `filters` contains the code needed to pre-process ECGs.  Specifically:

 * `filterFactory.py` contains a `FilterFactory` class that can perform a series of filters to a single lead of an ECG.

 * `ecgMatrix.py` contains a `ECGMatrix` class for an ECG matrix that can optionally apply filters to every lead of an ECG

 ## Pipelining and General Tools

 * `utils.py` contains a series of helper functions that can be used to do things like fetch ECGs, load ECGs from disk, apply class labels, apply filters, and split data into train/test sets.

 # Models

 ## Feature Extraction Based

 * `feature_extraction/ecgAutoencoder.py` contains the autoencoder model

 * `feature_extraction/autoencoder_train_automl.ipynb` is a notebook that trains the autoencoder and performs AutoML on the extracted features

 * `feature_extraction/baseline_features.py` contains functions to extract baseline features based on domain-specific knowledge

* `feature_extraction/data_prep.py` contains code to extract the baseline features and save to disk

 * `feature_extraction/model_traditional_feature_extraction.ipynb` is a notebook that trains does AutoML on the baseline features

 ## CNN

 * `model_development.ipynb` contains the code to train the CNN model

 ## ResNet CNN

 * The code for the ResNet model is in the `ResNet/` directory`

 ## External Validation

 The code for the external validation data is in the directory `external_validation/`.  `utils.py` contains the code to pull the cohort and diagnosis codes.

 The code used to load ECGs can not be shared, nor can the file with the connection info, schema names, or raw ECG files

 The code to identify labels, generate the cohort, and run predictions is in `external_validation/analysis.py`.