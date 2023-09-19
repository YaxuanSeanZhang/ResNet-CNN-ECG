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

#read test data as numpy
#y_test should be an array of 0/1
X_test = np.load("X_test_final.npy")
y_test = np.load("y_test_final.npy")


test_dataset = ImageDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

ECG_model = ECGNet()
if os.path.exists("ResNetmodel_final"): 
    print("Loading state dict...")
    ECG_model.load_state_dict(torch.load("ResNetmodel_final"))
    criterion = nn.BCELoss()
else:
    print('no model exists')
    
ECG_model.prediction(test_dataloader, criterion)


