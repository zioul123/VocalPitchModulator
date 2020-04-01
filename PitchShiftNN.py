import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

from tqdm.notebook import trange, tqdm

from IPython.display import HTML
import warnings

import torch
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

from Utils import *
    
class PitchShiftNN(nn.Module):
    """This neural network attempts to perform a direct pitch shift.
    
    """
    def __init__(self, n_input = 1026, n_hid = 513, n_output = 513):
        super().__init__()
        torch.manual_seed(0)

        self.input   = nn.Linear(n_input, n_hid)
        self.hidden  = nn.Linear(n_hid, n_hid)
        self.output  = nn.Linear(n_hid, n_output)
        self.relu    = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.relu(self.input(x))
        h2 = self.relu(self.hidden(h1))
        return self.relu(self.output(h2))

    def train_func(self, x, y, x_val, y_val, model, opt, loss_fn, epochs=10000, print_graph=False):
        print(x.shape)
        """Generic function for training a classifier.
        
        Args:
            x (torch.Tensor): The training data (N * d), may be on the GPU.
            y (torch.Tensor): The training labels (N * 1), may be on the GPU.
            x_val (torch.Tensor): The validation data (V * d), may be on the GPU.
            y_val (torch.Tensor): The validation data (V * 1), may be on the GPU.
            model (torch.nn.Module): The model to be trained, may be on the GPU.
            opt (torch.optim): The optimizer function.
            loss_fn (function): The loss function. 
            epochs (int): The number of epochs to perform.
            print_graph (boolean): Whether or not we want to store the loss/accuracy 
                per epoch, and plotting a graph.
        """
        # Function to calculate accuracy of model, for graphing purposes
        # def accuracy(y_hat, y):
        #    return torch.dist(y_hat, y, p=2)
        
        # Function/structures to log the current timestep
        if print_graph:
            loss_arr = []; acc_arr = []; val_acc_arr = [];
            
        def record_loss(_loss, _acc, _val_acc):
            loss_arr.append(_loss)
            acc_arr.append(_acc)
            val_acc_arr.append(_val_acc)
        
        # Actual training

        # Loop with progress bar
        for epoch in trange(epochs, desc='Training'):
            opt.zero_grad()
            y_hat = model(x)  
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            
            if print_graph:
                record_loss(loss.item(), 0, 0)
        
        if print_graph:
            plot_loss_graph(loss_arr=loss_arr, acc_arr=acc_arr, val_acc_arr=val_acc_arr)

        return loss.item()