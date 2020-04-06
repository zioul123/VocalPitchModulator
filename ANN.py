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

class TimbreEncoder(nn.Module):
    """This neural network attempts to identify a vowel, given an MFCC."""
    
    def __init__(self, n_mfcc=20, n_hid=12, n_timb=4, n_vowels=12):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            # nn.Linear(n_mfcc, n_timb), 
            nn.Linear(n_mfcc, n_mfcc), 
            nn.ReLU(), 
            nn.Linear(n_mfcc, n_hid), 
            nn.ReLU(), 
            nn.Linear(n_hid, n_timb), 
            nn.ReLU(),
            nn.Linear(n_timb, n_vowels),
            nn.Softmax())

    def forward(self, X):
        return self.net(X)

    def train_func(self, x, y, x_val, y_val, model, opt, loss_fn, epochs=10000, print_graph=False):
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
        def accuracy(y_hat, y):
            pred = torch.argmax(y_hat, dim=1)
            return (pred == y).float().mean()
        
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
                record_loss(loss.item(), accuracy(y_hat, y), accuracy(model(x_val),y_val))
        
        if print_graph:
            plot_loss_graph(loss_arr=loss_arr, acc_arr=acc_arr, val_acc_arr=val_acc_arr)

        return loss.item()
    
class TimbreVAE(nn.Module):
    """This neural network attempts to extract timbre features from MFCCs.
    
    It is a VAE that encodes the given MFCC to n_timb dimensions, and 
    attempts to decode the n_timb-vector to a n_mfcc vector to match the 
    original MFCC.
    """
    def __init__(self, n_mfcc=20, n_hid=12, n_timb=4):
        super().__init__()
        torch.manual_seed(0)

        self.n_mfcc  = n_mfcc

        self.en1     = nn.Linear(n_mfcc, n_hid)
        self.en_mu   = nn.Linear(n_hid, n_timb) 
        self.en_std  = nn.Linear(n_hid, n_timb) 
        self.de1     = nn.Linear(n_timb, n_hid)
        self.de2     = nn.Linear(n_hid, n_mfcc)
        self.relu    = nn.ReLU()
        self.Softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

    def encode(self, x):
        """Encodes a batch of samples."""
        h1 = self.relu(self.en1(x))
        return self.en_mu(h1), self.en_std(h1)

    def decode(self, z):
        """Decodes a batch of latent variables."""
        h2 = self.relu(self.de1(z))
        return self.sigmoid(self.de2(h2))
        # Use tanh because we want -1 to 1.
        # return self.tanh(self.de2(h2))

    def reparam(self, mu, logvar):
        """Reparameterization trick to sample z values."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

    def get_z(self, x):
        """Encode a batch of data points, x, into their z representations."""
        mu, logvar = self.encode(x)
        return self.reparam(mu, logvar)

    def train_func(self, x, x_val, model, opt, loss_fn, batch_size=128, epochs=10000, print_graph=False, desc='Training'):
        """Function to train a VAE.
        
        Args:
            x (torch.Tensor): The training data (N * d), may be on the GPU.
            x_val (torch.Tensor): The validation data (V * d), may be on the GPU.
            model (torch.nn.Module): The model to be trained, may be on the GPU.
            opt (torch.optim): The optimizer function.
            loss_fn (function): The loss function. 
            epochs (int): The number of epochs to perform.
            print_graph (boolean): Whether or not we want to store the loss/accuracy 
                per epoch, and plotting a graph.
        """
        if print_graph:
            loss_arr = []; val_loss_arr = []
        
        # Actual training

        n_batches = int(np.ceil(x.shape[0] / batch_size))
        batches = [ x[batch_idx * batch_size : ((batch_idx + 1) * batch_size 
                                                if (batch_idx < batch_size - 1) else
                                                x.shape[0])] 
                    for batch_idx in range(n_batches) ]

        # Loop with progress bar
        with trange(epochs, desc=desc) as t:
            for epoch in t:
                train_loss = 0
                for batch_idx, batch_x in enumerate(batches):
                    opt.zero_grad()
                    recon_x, mu, logvar = model(batch_x)
                    loss = loss_fn(recon_x, batch_x, mu, logvar)
                    loss.backward()
                    train_loss += loss.item()
                    opt.step()

                if print_graph:
                    loss_arr.append(train_loss / x.shape[0])

                    # Compute validation loss
                    recon_x, mu, logvar = model(x_val)
                    val_loss = (loss_fn(recon_x, x_val, mu, logvar)).item()
                    val_loss_arr.append(val_loss / x_val.shape[0])
                    # if (epoch % 1000 == 0):
                        # print("Val Loss:", val_loss_arr[-1])
                    # print("Loss arr:", loss_arr)
                t.set_postfix(loss=(train_loss / x.shape[0]))
                
        if print_graph:
            plot_loss_graph(loss_arr=loss_arr, val_loss_arr=val_loss_arr)

        # Compute validation loss
        recon_x, mu, logvar = model(x_val)
        val_loss = (loss_fn(recon_x, x_val, mu, logvar)).item() / x_val.shape[0]
        return train_loss / x.shape[0], val_loss

class TimbreMelDecoder(nn.Module):
    """This neural network attempts to recreate an FFT, given a mel spectrum and timbre vector"""
    
    def __init__(self, n_input=132, n_hid=260, n_hid2=386, n_ffts=513):
        super().__init__()
        torch.manual_seed(0)
        self.n_input  = n_input

        self.fc1     = nn.Linear(n_input, n_hid)
        self.fc2     = nn.Linear(n_hid, n_hid2)
        self.fc3     = nn.Linear(n_hid2, n_ffts)
        self.net     = nn.Sequential(self.fc1, 
                                     nn.ReLU(), 
                                     self.fc2, 
                                     nn.ReLU(), 
                                     self.fc3, 
                                     nn.ReLU())

        self.relu    = nn.ReLU()
        self.Softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

    def forward(self, X):
        return self.net(X)

    def train_func(self, x, y, x_val, y_val, model, opt, loss_fn, batch_size=128, epochs=10000, print_graph=False):
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
        loss_arr = []; 
        if print_graph:
            val_loss_arr = []

        # Actual training

        # ==============
        # # For batching
        # n_batches = int(np.ceil(x.shape[0] / batch_size))
        # batches = [ x[batch_idx * batch_size : ((batch_idx + 1) * batch_size 
        #                                         if (batch_idx < batch_size - 1) else
        #                                         x.shape[0])] 
        #             for batch_idx in range(n_batches) ]
        # labels = [ y[batch_idx * batch_size : ((batch_idx + 1) * batch_size 
        #                                         if (batch_idx < batch_size - 1) else
        #                                         y.shape[0])] 
        #             for batch_idx in range(n_batches) ]
        # ==============

        # Loop with progress bar
        with trange(epochs, desc='Training') as t:
            for epoch in t:
                # ==============
                # # For batching
                # train_loss = 0
                # total = 0
                # for batch_idx, batch_x in enumerate(batches):
                #     opt.zero_grad()
                #     y_hat = model(batch_x)
                #     loss = loss_fn(y_hat, y[batch_idx])
                #     # if (epoch % 100 == 0):
                #     #     print(y_hat.cpu().detach().numpy()[0])
                #     loss.backward()
                #     train_loss += loss.item()
                #     opt.step()
                # loss_arr.append(train_loss)
                # t.set_postfix(loss=train_loss)
                # ==============
               
                # ==============
                # # Non batching
                opt.zero_grad()
                y_hat = model(x)
                if (epoch % 100 == 0):
                    print(y_hat.cpu().detach().numpy()[0])
                loss = loss_fn(y_hat, y)
                loss.backward()
                opt.step()
                t.set_postfix(loss=loss.item())
                loss_arr.append(loss.item())
                # ==============

                if print_graph:
                    # Compute validation loss
                    y_hat = model(x_val)
                    val_loss = (loss_fn(y_hat, y_val)).item()
                    val_loss_arr.append(val_loss)
        
        if print_graph:
            plot_loss_graph(loss_arr=loss_arr, val_loss_arr=val_loss_arr)

        # Compute validation loss
        y_hat = model(x_val)
        val_loss = (loss_fn(y_hat, y_val)).item()
        return loss_arr[-1], val_loss
