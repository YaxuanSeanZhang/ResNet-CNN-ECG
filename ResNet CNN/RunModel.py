import pandas as pd
import numpy as np
import wfdb
import ast
import math, os

from image_dataset import ImageDataset, get_weighted_sampler
from torch.utils.data import DataLoader
from ECGNet import *

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

from filter import *
 
X_train = np.load("X_train_final.npy")
y_train = np.load("y_train_final.npy")
X_validation = np.load("X_validation_final.npy")
y_validation = np.load("y_validation_final.npy")
X_test = np.load("X_test_final.npy")
y_test = np.load("y_test_final.npy")

'''
#remove missing data
# Find all records that have at least one missing value in the 12-length arrays
missing_values_mask = np.any(np.isnan(X_train), axis=(1,2)) | np.any(np.apply_along_axis(lambda x: len(x)==0, axis=2, arr=X_train), axis=1)

# Get the indices of the records that have missing values
missing_values_indices = np.where(missing_values_mask)[0]
X_train = X_train[~missing_values_mask]
y_train = y_train[~missing_values_mask]

#print("Indices of removed records:", missing_values_indices)
'''

#filter
#X_train, y_train = ecg_preprocess_pipeline(X_train, y_train,normalize=False)
#np.save("X_train_filter", X_train)
#np.save("y_train_filter", y_train)


train_dataset = ImageDataset(X_train, y_train)

'''
# Find all records that have at least one missing value in the 12-length arrays
missing_values_mask = np.any(np.isnan(X_test), axis=(1,2)) | np.any(np.apply_along_axis(lambda x: len(x)==0, axis=2, arr=X_test), axis=1)

# Get the indices of the records that have missing values
missing_values_indices = np.where(missing_values_mask)[0]
X_test = X_test[~missing_values_mask]
y_test = y_test[~missing_values_mask]
'''

#train_sampler = get_weighted_sampler(y_train, num_samples=1500)
train_dataloader = DataLoader(train_dataset, batch_size=256,shuffle=True)

#X_test, y_test = ecg_preprocess_pipeline(X_test, y_test,normalize=False)
#np.save("X_test_filter", X_test)
#np.save("y_test_filter", y_test)

validation_dataset = ImageDataset(X_validation, y_validation)
validation_dataloader = DataLoader(validation_dataset, batch_size=256, shuffle=True)

test_dataset = ImageDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

ECG_model = ECGNet()
if os.path.exists("ResNetmodel_final"): 
    print("Loading state dict...")
    ECG_model.load_state_dict(torch.load("ResNetmodel_final"))
    criterion = nn.BCELoss()
else:
    criterion = nn.BCELoss()

    optimizer_ECG = torch.optim.Adam(ECG_model.parameters(), lr=1e-3)

    fitresult_ECG = ECG_model.fit(train_dataloader, validation_dataloader, criterion, optimizer_ECG)

    #BASE_model = BaseNet()

    #optimizer_BASE = torch.optim.Adam(BASE_model.parameters(), lr=1e-3)

    #fitresult_BASE = BASE_model.fit(train_dataloader, test_dataloader, criterion, optimizer_BASE)
    print("Saving state dict...")
    torch.save(ECG_model.state_dict(), "ResNetmodel_final")

ECG_model.prediction(test_dataloader, criterion)


