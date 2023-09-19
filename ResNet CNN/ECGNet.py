import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np

import torch
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def conv_2d(in_planes, out_planes, stride=(1,1), size=3):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,size), stride=stride,
                    padding=(0,(size-1)//2), 
                    bias=False)

class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1,1), downsample=None, size=3, res=True):
        super(BasicBlock2d, self).__init__()
        self.conv = conv_2d(inplanes, planes, stride, size=size)
        #self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=(1,4))
        #self.conv2 = conv_2d(planes, planes, size=size)
        #self.bn2 = nn.BatchNorm2d(planes)
        #self.conv3 = conv_2d(planes, planes, size=size)
        #self.bn3 = nn.BatchNorm2d(planes)
        #self.dropout = nn.Dropout(.3)
        self.downsample = downsample
        self.stride = stride
        self.res = res

    def forward(self, x):
        residual = x

        #out = self.bn1(x)
        out = self.conv(x)
        out = self.relu(out)
        #out = self.bn2(out)
        #out = self.relu(out)
        #out = self.conv2(out) 
        #out = self.bn3(out)
        #out = self.relu(out)
        #out = self.conv3(out) 

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out = out + residual
        #out = self.relu(out)

        #out = self.maxpool(out)
        #out = self.dropout(out)
        
        return out
    
class ECGNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=1):#, layers=[2, 2, 2, 2, 2, 2]
        super(ECGNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        #self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(30,6), stride=(2,1), padding=(0,0),
        #                       bias=False)
        
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, (12,30), (1, 1)),
                nn.ReLU()
            )
        
        self.bn1 = nn.BatchNorm2d(32)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,10))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,4))
        
        self.layers1_list = nn.ModuleList()
        #self.layers2_list = nn.ModuleList()
        self.inplanes = 32 
        self.dropout = nn.Dropout(.6)

        self.layers1 = nn.Sequential()
        self.layers1.add_module('layer3*3_1_1', self._make_layer2d(BasicBlock2d, 32, 2, stride=(1,1), size=3))
        #self.layers1.add_module('layer3*3_1_2', self._make_layer2d(BasicBlock2d, 32, 2, stride=(1,1), size=3))
        self.layers1_list.append(self.layers1)

        #self.inplanes = 32 
        self.layers1 = nn.Sequential()
        self.layers1.add_module('layer5*5_1_1', self._make_layer2d(BasicBlock2d, 32, 2, stride=(1,1), size=5))
        #self.layers1.add_module('layer5*5_1_2', self._make_layer2d(BasicBlock2d, 32, 2, stride=(1,1), size=5))
        self.layers1_list.append(self.layers1)

        #self.inplanes = 32 
        self.layers1 = nn.Sequential()
        self.layers1.add_module('layer9*9_1_1', self._make_layer2d(BasicBlock2d, 32, 2, stride=(1,1), size=9))
        #self.layers1.add_module('layer7*7_1_2', self._make_layer2d(BasicBlock2d, 32, 2, stride=(1,1), size=7))
        self.layers1_list.append(self.layers1)

        self.conv3 = nn.Sequential(
                nn.Conv2d(32,16, (1,4), (1, 1)),
                nn.ReLU()
            )

        self.maxpool3 = nn.MaxPool2d(kernel_size=(1,2))
        
        self.fc = nn.Sequential(
                nn.Linear(2880, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )


    def _make_layer2d(self, block, planes, blocks, stride=(1,2), size=3, res=True):
        downsample = None
        #make sure the dimension matches
        if stride != (1,1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=(1,1), padding=(0,0), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))

        return nn.Sequential(*layers)
    

    def forward(self, x0):
        x0 = x0.unsqueeze(1)
        x0 = self.conv1(x0)
        #x0 = self.bn1(x0)
        #x0 = self.relu(x0)
        x0 = self.maxpool1(x0)
        x0 = self.dropout(x0)

        xs = []
        x = self.layers1_list[0](x0)
        x = self.maxpool2(x)
        x = self.dropout(x)
        xs.append(x)

        x = self.layers1_list[1](x0)
        x = self.maxpool2(x)
        x = self.dropout(x)
        xs.append(x)

        x = self.layers1_list[2](x0)
        x = self.maxpool2(x)
        x = self.dropout(x)
        xs.append(x)

        out = torch.cat(xs, dim=2)

        out = self.conv3(out)     
        #x0 = self.relu(x0)
        out = self.maxpool3(out)
        out = self.dropout(out)

        out = out.view(out.size(0), -1)
        #out = torch.cat([out, fr], dim=1)
        out = self.fc(out)
        out = torch.nn.Sigmoid()(out)

        return out
    
    def fit(self, train_loader, validation_loader, criterion, optimizer):
        
        # Epoch loop
        for i in range(30):
            print(f'\n===== EPOCH {i} =====')

            training_loss = 0
            test_loss = 0
            error_rate = 0
            n_samples = 0

            self.train()
            for j,(image,label) in tqdm(enumerate(train_loader)):
                # Forward pass (consider the recommmended functions in homework writeup)
                image = torch.transpose(image,1,2)
                label = label.reshape(-1, 1)
                output = self.forward(image)

                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the loss and error rate
                training_loss += loss.item()
                pred = (output>0.5).to(torch.float32).reshape(-1, 1)
                error_rate += (pred != label).sum().item()
                n_samples += len(label) 

            error_rate_test = 0
            n_samples_test = 0

            self.eval()
            for j,(image,label) in tqdm(enumerate(validation_loader)):   
                image = torch.transpose(image,1,2)
                label = label.reshape(-1, 1)
                output = self.forward(image)
                loss = criterion(output, label)

                # Track the loss and error rate
                test_loss += loss.item()
                pred = (output>0.5).to(torch.float32).reshape(-1, 1)
                error_rate_test += (pred != label).sum().item()
                n_samples_test += len(label) 
                
            # Print/return training loss and error rate in each epoch
            print('training loss for each epoch is:', training_loss)
            print('training error rate for each epoch is:', error_rate/n_samples)
            print('validation loss for each epoch is:', test_loss)
            print('validation error rate for each epoch is:', error_rate_test/n_samples_test)
        
    def prediction(self, test_loader, criterion):
        pred_list = []
        label_list = []
        output_list = []
        error_rate = 0
        n_samples = 0
        test_loss = 0

        for j,(image,label) in tqdm(enumerate(test_loader)):   
                image = torch.transpose(image,1,2)
                label = label.reshape(-1, 1)
                output = self.forward(image)
                loss = criterion(output, label)

                # Track the loss and error rate
                test_loss += loss.item()
                pred = (output>0.5).to(torch.float32).reshape(-1, 1)
                error_rate += (pred != label).sum().item()
                n_samples += len(label) 

                output_list.append([i.item() for i in output])
                pred_list.append([i.item() for i in pred])
                label_list.append([i.item() for i in label])
        
        print('test loss for each epoch is:', test_loss)
        print('test error rate for each epoch is:', error_rate/n_samples)

        #AUC
        # y_pred is an array of predicted probabilities (e.g., output of predict_proba())
        # y_test is an array of true labels (0 or 1)
        auc = roc_auc_score(np.concatenate(label_list), np.concatenate(output_list))
        print('AUC:', auc)

        # Print the confusion matrix
        cm = confusion_matrix(np.concatenate(label_list), np.concatenate(pred_list))
        labels = ['Non-CD', 'CD']

        # Plot the confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels, yticklabels=labels,
            title='Confusion matrix',
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over the data and create a text annotation for each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2. else "black")
        fig.tight_layout()
        plt.show()

class BaseNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=1):#, layers=[2, 2, 2, 2, 2, 2]
        super(BaseNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, (30,12), (1, 1)),
                nn.ReLU()
            )
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(10,1))
        
        self.dropout = nn.Dropout(.6)

        self.conv2 = nn.Sequential(
                nn.Conv2d(32,64, (3,1), (1, 1)),
                nn.ReLU()
            )

        self.maxpool2 = nn.MaxPool2d(kernel_size=(4,1))
 
        self.conv3 = nn.Sequential(
                nn.Conv2d(64,16, (4,1), (1, 1)),
                nn.ReLU()
            )

        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,1))

        self.fc = nn.Sequential(
                nn.Linear(320, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
    

    def forward(self, x0):
        x0 = x0.unsqueeze(1)
        x0 = self.conv1(x0)     
        #x0 = self.relu(x0)
        x0 = self.maxpool1(x0)
        x0 = self.dropout(x0)

        x0 = self.conv2(x0)     
        #x0 = self.relu(x0)
        x0 = self.maxpool2(x0)
        x0 = self.dropout(x0)

        x0 = self.conv3(x0)     
        #x0 = self.relu(x0)
        x0 = self.maxpool3(x0)
        x0 = self.dropout(x0)

        x0 =x0.view(x0.size(0), -1)
        #x0 = torch.flatten(x0)
        #out = torch.cat([out, fr], dim=1)
        x0 = self.fc(x0)
        #x0 = self.relu(x0)
        x0 = torch.nn.Sigmoid()(x0)

        return x0
    
    def fit(self, train_loader, test_loader, criterion, optimizer):
        
        # Epoch loop
        for i in range(30):
            print(f'\n===== EPOCH {i} =====')

            training_loss = 0
            test_loss = 0
            error_rate = 0
            n_samples = 0

            self.train()
            for j,(image,label) in tqdm(enumerate(train_loader)):
                # Forward pass (consider the recommmended functions in homework writeup)
                #image = torch.transpose(image,1,2)
                #image = torch.reshape(image,(32,5000,12,1))
                label = label.reshape(-1, 1)
                output = self.forward(image)

                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the loss and error rate
                training_loss += loss.item()
                pred = (output>0.5).to(torch.float32).reshape(-1, 1)
                error_rate += (pred != label).sum().item()
                n_samples += len(label) 

            error_rate = 0
            n_samples = 0

            self.eval()
            for j,(image,label) in tqdm(enumerate(test_loader)):   
                #image = torch.transpose(image,1,2)
                label = label.reshape(-1, 1)
                output = self.forward(image)
                loss = criterion(output, label)

                # Track the loss and error rate
                test_loss += loss.item()
                pred = (output>0.5).to(torch.float32).reshape(-1, 1)
                error_rate += (pred != label).sum().item()
                n_samples += len(label) 
                
            # Print/return training loss and error rate in each epoch
            print('training loss for each epoch is:', training_loss)
            print('test loss for each epoch is:', test_loss)
            print('error rate for each epoch is:', error_rate/n_samples)
        
        pred_list = []
        label_list = []
        for j,(image,label) in tqdm(enumerate(test_loader)):   
                image = torch.transpose(image,1,2)
                label = label.reshape(-1, 1)
                output = self.forward(image)
                loss = criterion(output, label)

                # Track the loss and error rate
                test_loss += loss.item()
                pred = (output>0.5).to(torch.float32).reshape(-1, 1)
                error_rate += (pred != label).sum().item()
                n_samples += len(label) 

                pred_list.append(pred)
                label_list.append(label)

        # Print the confusion matrix
        cm = confusion_matrix(label_list, pred_list)
        labels = ['Non-CD', 'CD']

        # Plot the confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels, yticklabels=labels,
            title='Confusion matrix',
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over the data and create a text annotation for each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2. else "black")
        fig.tight_layout()
        plt.show()


