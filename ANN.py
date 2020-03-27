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

class TimbreEncoder(nn.Module):
    
    def __init__(self, n_mfcc=20, n_hid1=12, n_timb=4, n_vowels=12):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            # nn.Linear(n_mfcc, n_timb), 
            nn.Linear(n_mfcc, n_mfcc), 
            nn.ReLU(), 
            nn.Linear(n_mfcc, n_hid1), 
            nn.ReLU(), 
            nn.Linear(n_hid1, n_timb), 
            nn.ReLU(),
            nn.Linear(n_timb, n_vowels),
            nn.Softmax())

    def forward(self, X):
        return self.net(X)
    
def fit(x, y, x_val, y_val, model, opt, loss_fn, epochs=10000, print_graph=False):
    """Generic function for training a model.
    
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
    
    #function to calculate accuracy of model, for graphing purposes
    def accuracy(y_hat, y):
        pred = torch.argmax(y_hat, dim=1)
        return (pred == y).float().mean()
    
    if print_graph:
        loss_arr = []; acc_arr = []; val_acc_arr = [];
    
    # Loop with progress bar
    for epoch in trange(epochs, desc='Training'):
        
        #compute the predicted distribution
        y_hat = model(x)  
        
        loss = loss_fn(y_hat, y) 
        loss.backward()
        
        if print_graph:
            loss_arr.append(loss.item())
            acc_arr.append(accuracy(y_hat, y))
            val_acc_arr.append(accuracy(model(x_val),y_val))

        opt.step()
        opt.zero_grad()
    
    if print_graph:
        plt.figure(figsize=(15, 10))
        plt.plot(loss_arr, 'r-', label='loss')
        plt.plot(acc_arr, 'b-', label='train accuracy')
        plt.plot(val_acc_arr, 'g-', label='val accuracy')
        plt.title("Loss plot")
        plt.xlabel("Epoch")
        plt.legend(loc='best')
        plt.show()
        print('Loss before/after: {}, {}'
              .format(loss_arr[0], loss_arr[-1]))
        print('Training accuracy before/after: {}, {}'
              .format(acc_arr[0], acc_arr[-1]))
        print('Validation accuracy before/after: {}, {}'
              .format(val_acc_arr[0], val_acc_arr[-1]))
                
        # pred = torch.argmax(y_hat, dim=1)
        # plt.scatter(x.cpu().detach().numpy()[:,0], 
        #             x.cpu().detach().numpy()[:,1], 
        #             c=pred.cpu().detach().numpy()) #, cmap=my_cmap)
        # plt.show()

    return loss.item()
