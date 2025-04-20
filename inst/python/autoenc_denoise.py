# -*- coding: utf-8 -*-
"""
Denoising Autoencoder (Autoencoder_Denoise AE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from random import sample

torch.set_grad_enabled(True)

class Autoencoder_Denoise_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.data[index]


class DNS_Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(DNS_Autoencoder, self).__init__()
        
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


#Create denoising autoencoder (Autoencoder_Denoise AE)
def add_noise(data, noise_factor=0.3):
    noisy = torch.randn_like(data)
    noisy = data + noisy * noise_factor
    
    return noisy


def autoenc_denoise_create(input_size, encoding_size, noise_factor=0.3):
    input_size = int(input_size)
    encoding_size = int(encoding_size)
    
    dns = DNS_Autoencoder(input_size, encoding_size)
    dns.float()
    dns.noise_factor=noise_factor
    
    return dns


# Train the autoencoder
def autoenc_denoise_train(dns, data, batch_size=32, num_epochs = 1000, learning_rate = 0.001):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(dns.parameters(), lr=learning_rate)

  train_loss = []
  val_loss = []
  
  noise_factor = dns.noise_factor
  
  for epoch in range(num_epochs):
      # Train Test Split
      array = data.to_numpy()
      array = array[:, :]
      
      val_sample = sample(range(1, data.shape[0], 1), k=int(data.shape[0]*0.3))
      train_sample = [v for v in range(1, data.shape[0], 1) if v not in val_sample]
      
      train_data = array[train_sample, :]
      val_data = array[val_sample, :]
      
      ds_train = Autoencoder_Denoise_TS(train_data)
      ds_val = Autoencoder_Denoise_TS(val_data)
      train_loader = DataLoader(ds_train, batch_size=batch_size)
      val_loader = DataLoader(ds_val, batch_size=batch_size)
    
      # Train
      train_epoch_loss = []
      val_epoch_loss = []
      dns.train()
      for train_data in train_loader:
          train_input, _ = train_data
          #Add noise to train data
          train_input = add_noise(train_input, noise_factor=noise_factor)
          train_input = train_input.float()
          optimizer.zero_grad()
          train_output = dns(train_input)
          train_batch_loss = criterion(train_output, train_input)
          train_batch_loss.backward()
          optimizer.step()
          train_epoch_loss.append(train_batch_loss.item())
          
          
      # Validation
      dns.eval()
      for val_data in val_loader:
          val_input, _ = val_data
          #Add noise to validation data
          val_input = add_noise(val_input, noise_factor=noise_factor)
          val_input = val_input.float()
          val_output = dns(val_input)
          val_batch_loss = criterion(val_output, val_input)
          val_epoch_loss.append(val_batch_loss.item())
          
      train_loss.append(np.mean(train_epoch_loss))
      val_loss.append(np.mean(val_epoch_loss))
  
  return dns, np.array(train_loss), np.array(val_loss)  


def autoenc_denoise_fit(dns, data, batch_size = 32, num_epochs = 1000, learning_rate = 0.001):
  batch_size = int(batch_size)
  num_epochs = int(num_epochs)
  
  dns = autoenc_denoise_train(dns, data, batch_size = batch_size, num_epochs = num_epochs, learning_rate = 0.001)
  return dns


def autoenc_denoise_encode_data(dns, data_loader):
  # Encode the synthetic time series data using the trained autoencoder
  encoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      inputs = inputs.view(inputs.size(0), -1)
      encoded = dns.encoder(inputs)
      encoded_data.append(encoded.detach().numpy())

  encoded_data = np.concatenate(encoded_data, axis=0)

  return encoded_data


def autoenc_denoise_encode(dns, data, batch_size = 32):
  array = data.to_numpy()
  array = array[:, :]
  
  ds = Autoencoder_Denoise_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_data = autoenc_denoise_encode_data(dns, train_loader)
  
  return(encoded_data)


def autoenc_denoise_encode_decode_data(dns, data_loader):
  # Encode the synthetic time series data using the trained autoencoder
  encoded_decoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      inputs = inputs.view(inputs.size(0), -1)
      encoded = dns.encoder(inputs)
      decoded = dns.decoder(encoded)
      encoded_decoded_data.append(decoded.detach().numpy())

  encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)

  return encoded_decoded_data


def autoenc_denoise_encode_decode(dns, data, batch_size = 32):
  array = data.to_numpy()
  array = array[:, :, np.newaxis]
  
  ds = Autoencoder_Denoise_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_decoded_data = autoenc_denoise_encode_decode_data(dns, train_loader)
  
  return(encoded_decoded_data)
