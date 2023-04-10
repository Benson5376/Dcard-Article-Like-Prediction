import numpy as np
import pandas as pd
from pandas import read_csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim

class MLP(nn.Module):
    
    def __init__(self):
        super(MLP,self).__init__()
        
        self.linear1=torch.nn.Linear(16, 100)
        self.relu = torch.nn.LeakyReLU()
        
        self.linear2=torch.nn.Linear(100, 30)
        self.relu2 = torch.nn.LeakyReLU()
        
        self.linear3=torch.nn.Linear(30, 10)
        self.relu3 = torch.nn.LeakyReLU()
        
        self.linear4=torch.nn.Linear(10, 1)
        self.relu4 = torch.nn.LeakyReLU()
        
    def forward(self, x):
        x=self.linear1(x)
        x=self.relu(x)
        
        x=self.linear2(x)
        x=self.relu2(x)
        
        x=self.linear3(x)
        x=self.relu3(x)
        
        x=self.linear4(x)
        x=self.relu4(x)
        return x
    
    
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x