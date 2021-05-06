import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pickle


class LabelToImageNet(nn.Module):
    def __init__(self, output_size=28*28):
        super().__init__()
        
        self.hidden_1 = nn.Linear(1, 20)
        self.hidden_2 = nn.Linear(20, 10)
        self.hidden_3 = nn.Linear(10, 10)
        self.hidden_4 = nn.Linear(10, 250)
        self.hidden_5 = nn.Linear(250, 500)
        self.hidden_6 = nn.Linear(500, 28*28)
        #self.hidden_3 = nn.Linear(500, 250)
        #self.hidden_4 = nn.Linear(250, 28*28)
        #self.hidden_5 = nn.Linear(*16, 28*28)        
    
    def forward(self, x):
        #x = torch.Tensor([[int(i == z) for i in range(10)] for z in x]).to(device).float()
        x = self.hidden_1(x)
        x =F.relu(x)
        x = self.hidden_2(x)
        x = F.relu(x)
        x = self.hidden_3(x)
        x = F.relu(x)
        x = self.hidden_4(x)
        x = F.relu(x)
        x = self.hidden_5(x)
        x = F.relu(x)
        x = self.hidden_6(x)
        #x = F.relu(x)
        x = torch.sigmoid(x)
        return x


def get_sequential_model():
    return nn.Sequential(
        nn.Linear(1, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 250),
        nn.ReLU(),
        nn.Linear(250, 500),
        nn.ReLU(),
        nn.Linear(500, 28*28),
        nn.Sigmoid()
    )
