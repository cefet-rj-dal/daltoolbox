# -*- coding: utf-8 -*-
"""
Denoising Autoencoder (DNS AE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class DNS_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.data[index]


class DNS_AE(nn.Module):
    ##NN architecture for the autoencoder
    def __init__(self, input_size, encoding_size):
        super(DNS_AE, self).__init__()
        
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


#Create denoising autoencoder (DNS AE)
def add_noise(data, noise_factor=0.3):
    noisy = torch.randn_like(data)
    noisy = data + noisy * noise_factor
    #noisy = torch.clip(noisy, 0., .1)
    
    return noisy


def dns_ae_create(input_size, encoding_size):
    input_size = int(input_size)
    encoding_size = int(encoding_size)
    
    dae = DNS_AE(input_size, encoding_size)
    dae.float()
    
    return dae


#Train DNS AE
def dns_ae_train(dae, train_loader, val_loader, num_epochs = 1000, learning_rate = 0.001, noise_factor=0.3, return_loss=False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dae.parameters(), lr=learning_rate)
    
    train_loss = []
    val_loss = []
    
    for epoch in range(num_epochs):
        train_epoch_loss = []
        val_epoch_loss = []
        # Train
        dae.train()
        for train_data in train_loader:
            train_input, _ = train_data
            train_input = train_input.float()
            optimizer.zero_grad()
            train_output = dae(train_input)
            train_batch_loss = criterion(train_output, train_input)
            train_batch_loss.backward()
            optimizer.step()
            train_epoch_loss.append(train_batch_loss.item())
            
            
        # Validation
        dae.eval()
        for val_data in val_loader:
            val_input, _ = val_data
            val_input = val_input.float()
            val_output = dae(val_input)
            val_batch_loss = criterion(val_output, val_input)
            val_epoch_loss.append(val_batch_loss.item())
            
        train_loss.append(np.mean(train_epoch_loss))
        val_loss.append(np.mean(val_epoch_loss))
  
    if return_loss:
      return dae, train_loss, val_loss
    else:
      return dae


def dns_ae_fit(dae, data, batch_size = 32, num_epochs = 1000, learning_rate = 0.001, noise_factor=0.3, return_loss=False):
    batch_size = int(batch_size)
    num_epochs = int(num_epochs)
    
    array = data.to_numpy()
    array = array[:, :]
    
    val_sample = sample(range(1, data.shape[0], 1), k=int(data.shape[0]*0.3))
    train_sample = [v for v in range(1, data.shape[0], 1) if v not in val_sample]
    
    train_data = array[train_sample, :]
    val_data = array[val_sample, :]
    
    ds_train = DNS_TS(train_data)
    ds_val = DNS_TS(val_data)
    train_loader = DataLoader(ds_train, batch_size=batch_size)
    val_loader = DataLoader(ds_val, batch_size=batch_size)
    
    if return_loss:
      dae, train_loss, val_loss = dns_ae_train(dae, train_loader, val_loader, num_epochs = num_epochs, learning_rate = 0.001, return_loss=return_loss)
      return dae, train_loss, val_loss
    else:
      dae = dns_ae_train(dae, train_loader, val_loader, num_epochs = num_epochs, learning_rate = 0.001, return_loss=return_loss)
      return dae


def dns_encode_data(autoencoder, data_loader):
    # Encode the synthetic time series data using the trained autoencoder
    encoded_data = []
    for data in data_loader:
        inputs, _ = data
        #Add noise to data before train
        inputs = add_noise(inputs)
        inputs = inputs.float()
        inputs = inputs.view(inputs.size(0), -1)
        encoded = autoencoder.encoder(inputs)
        encoded_data.append(encoded.detach().numpy())
        
    encoded_data = np.concatenate(encoded_data, axis=0)
    
    return encoded_data


def dns_encode(autoencoder, data, batch_size = 32):
    array = data.to_numpy()
    array = array[:, :]
    
    ds = DNS_TS(array)
    train_loader = DataLoader(ds, batch_size=batch_size)
    
    encoded_data = dns_encode_data(autoencoder, train_loader)
    
    return(encoded_data)


def dns_encode_decode_data(autoencoder, data_loader):
    # Encode the synthetic time series data using the trained autoencoder
    encoded_decoded_data = []
    for data in data_loader:
        inputs, _ = data
        #Add noise to data before train
        inputs = add_noise(inputs)
        inputs = inputs.float()
        inputs = inputs.view(inputs.size(0), -1)
        encoded = autoencoder.encoder(inputs)
        decoded = autoencoder.decoder(encoded)
        encoded_decoded_data.append(decoded.detach().numpy())
        
    encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)
    
    return encoded_decoded_data


def dns_encode_decode(autoencoder, data, batch_size = 32):
    array = data.to_numpy()
    array = array[:, :]
    
    ds = DNS_TS(array)
    train_loader = DataLoader(ds, batch_size=batch_size)
    
    encoded_decoded_data = dns_encode_decode_data(autoencoder, train_loader)
    
    return(encoded_decoded_data)
