# -*- coding: utf-8 -*-
"""
Stacked Autoencoder (SAE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import sample


class SAE_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.data[index]


class SAE(nn.Module):
    ##NN architecture for the autoencoder
    def __init__(self, input_size, encoding_size):
        super(SAE, self).__init__()
        
        #Encode NN
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64), #NN Input layer
            nn.ReLU(True), #NN Hidden layer
            nn.Linear(64, encoding_size) ##NN Output layter
            )
        
        #Decode NN
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 64), #NN Input layer
            nn.ReLU(True), #NN Hidden layer
            nn.Linear(64, input_size) ##NN Output layter
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x


#Create Stack of Autoencoders
def sae_create(input_size, encoding_size, k=3):
    input_size = int(input_size)
    encoding_size = int(encoding_size)
    k = int(k)
    
    #Stack of k autoencoders
    stack = []
    for k in range(k):
        stack.append(SAE(input_size, encoding_size))
        stack[k].float()
    
    return stack


#Train SAE
def sae_train(sae, data, batch_size=32, num_epochs = 1000, learning_rate = 0.001):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(sae.parameters(), lr=learning_rate)

  train_loss = []
  val_loss = []
  
  for epoch in range(num_epochs):
      # Train Test Split
      array = data.to_numpy()
      array = array[:, :]
      
      val_sample = sample(range(1, data.shape[0], 1), k=int(data.shape[0]*0.3))
      train_sample = [v for v in range(1, data.shape[0], 1) if v not in val_sample]
      
      train_data = array[train_sample, :]
      val_data = array[val_sample, :]
      
      ds_train = SAE_TS(train_data)
      ds_val = SAE_TS(val_data)
      train_loader = DataLoader(ds_train, batch_size=batch_size)
      val_loader = DataLoader(ds_val, batch_size=batch_size)
    
      # Train
      train_epoch_loss = []
      val_epoch_loss = []
      sae.train()
      for train_data in train_loader:
          train_input, _ = train_data
          train_input = train_input.float()
          optimizer.zero_grad()
          train_output = sae(train_input)
          train_batch_loss = criterion(train_output, train_input)
          train_batch_loss.backward()
          optimizer.step()
          train_epoch_loss.append(train_batch_loss.item())
          
          
      # Validation
      sae.eval()
      for val_data in val_loader:
          val_input, _ = val_data
          val_input = val_input.float()
          val_output = sae(val_input)
          val_batch_loss = criterion(val_output, val_input)
          val_epoch_loss.append(val_batch_loss.item())
          
      train_loss.append(np.mean(train_epoch_loss))
      val_loss.append(np.mean(val_epoch_loss))
  
  sae.train_loss = train_loss
  sae.val_loss = val_loss
  return sae


def sae_fit(stack, data, batch_size = 32, num_epochs = 1000, learning_rate = 0.001):
    batch_size = int(batch_size)
    num_epochs = int(num_epochs)

    #STEP 1 - Start fitting first k autoencoder using original input layer        
    ae_k1 = stack[0]
    ae_k1 = sae_train(ae_k1, data, num_epochs = num_epochs, learning_rate = learning_rate)
    ae_k_out = sae_encode_decode(ae_k1, data)
    
    #STEP 2 - Fit internal layers using outputs from previous layers
    internal = int(len(stack)-1)
    
    for k in range(1, internal):        
        ae_k = stack[k]
        ae_k = sae_train(ae_k, data, num_epochs = num_epochs, learning_rate = learning_rate)
        ae_k_out = sae_encode_decode(ae_k, ae_k_out)
    
    #STEP 3 - Fit last k autoencoder
    #print(f'Fit ae_k{len(stack)-1}')    
    sae = stack[-1]
    sae = sae_train(ae_k, data, num_epochs = num_epochs, learning_rate = 0.001)
    
    return sae


def sae_encode_data(sae, data_loader):
    # Encode the synthetic time series data using the trained autoencoder
    encoded_data = []
    for data in data_loader:
        inputs, _ = data
        inputs = inputs.float()
        inputs = inputs.view(inputs.size(0), -1)
        encoded = sae.encoder(inputs)
        encoded_data.append(encoded.detach().numpy())
        
    encoded_data = np.concatenate(encoded_data, axis=0)
    
    return encoded_data


def sae_encode(sae, data, batch_size = 32):
    #Condition to check numpy array in internal stacked autoencoders
    if not isinstance(data, np.ndarray):
        array = data.to_numpy()
    else:
        array = data
        
    array = array[:, :]
    
    ds = SAE_TS(array)
    train_loader = DataLoader(ds, batch_size=batch_size)
    
    encoded_data = sae_encode_data(sae, train_loader)
    
    return(encoded_data)


def sae_encode_decode_data(sae, data_loader):
    # Encode the synthetic time series data using the trained autoencoder
    encoded_decoded_data = []
    for data in data_loader:
        inputs, _ = data
        inputs = inputs.float()
        inputs = inputs.view(inputs.size(0), -1)
        encoded = sae.encoder(inputs)
        decoded = sae.decoder(encoded)
        encoded_decoded_data.append(decoded.detach().numpy())
        
    encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)
    
    return encoded_decoded_data


def sae_encode_decode(sae, data, batch_size = 32):
    #Condition to check numpy array in internal stacked autoencoders
    if not isinstance(data, np.ndarray):
        array = data.to_numpy()
    else:
        array = data
    
    array = array[:, :]
    
    ds = SAE_TS(array)
    train_loader = DataLoader(ds, batch_size=batch_size)
    
    encoded_decoded_data = sae_encode_decode_data(sae, train_loader)
    
    return(encoded_decoded_data)
