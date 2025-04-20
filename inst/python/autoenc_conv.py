import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import sample


class Autoencoder_Conv_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.data[index]

class Autoencoder_Conv(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder_Conv, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64, encoding_size))
            
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 64),
            nn.LeakyReLU(),
            nn.Unflatten(1, (64, 1)),
            nn.ConvTranspose1d(64, input_size, kernel_size=1),
            nn.Sigmoid()
            )
    
    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x

    
# Create the cae
def autoenc_conv_create(input_size, encoding_size):
  input_size = int(input_size)
  encoding_size = int(encoding_size)
  
  cae = Autoencoder_Conv(input_size, encoding_size)
  cae = cae.float()
  return cae  

# Train the cae
def autoenc_conv_train(cae, data, batch_size=32, num_epochs = 1000, learning_rate = 0.001):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(cae.parameters(), lr=learning_rate)

  train_loss = []
  val_loss = []
  
  for epoch in range(num_epochs):
      # Train Test Split
      array = data.to_numpy()
      array = array[:, :, np.newaxis]
      
      val_sample = sample(range(1, data.shape[0], 1), k=int(data.shape[0]*0.3))
      train_sample = [v for v in range(1, data.shape[0], 1) if v not in val_sample]
      
      train_data = array[train_sample, :]
      val_data = array[val_sample, :]
      
      ds_train = Autoencoder_Conv_TS(train_data)
      ds_val = Autoencoder_Conv_TS(val_data)
      train_loader = DataLoader(ds_train, batch_size=batch_size)
      val_loader = DataLoader(ds_val, batch_size=batch_size)
    
      # Train
      train_epoch_loss = []
      val_epoch_loss = []
      cae.train()
      for train_data in train_loader:
          train_input, _ = train_data
          train_input = train_input.float()
          optimizer.zero_grad()
          train_output = cae(train_input)
          train_batch_loss = criterion(train_output, train_input)
          train_batch_loss.backward()
          optimizer.step()
          train_epoch_loss.append(train_batch_loss.item())
          
          
      # Validation
      cae.eval()
      for val_data in val_loader:
          val_input, _ = val_data
          val_input = val_input.float()
          val_output = cae(val_input)
          val_batch_loss = criterion(val_output, val_input)
          val_epoch_loss.append(val_batch_loss.item())
          
      train_loss.append(np.mean(train_epoch_loss))
      val_loss.append(np.mean(val_epoch_loss))
  
  return cae, np.array(train_loss), np.array(val_loss)  


def autoenc_conv_fit(cae, data, batch_size = 32, num_epochs = 1000, learning_rate = 0.001):
  batch_size = int(batch_size)
  num_epochs = int(num_epochs)

  cae = autoenc_conv_train(cae, data, batch_size=batch_size, num_epochs = num_epochs, learning_rate = 0.001)
  return cae


def autoenc_conv_encode_data(cae, data_loader):
  # Encode the synthetic time series data using the trained cae
  encoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = cae.encoder(inputs)
      encoded_data.append(encoded.detach().numpy())

  encoded_data = np.concatenate(encoded_data, axis=0)

  return encoded_data

def autoenc_conv_encode(cae, data, batch_size = 32):
  array = data.to_numpy()
  array = array[:, :, np.newaxis]
  
  ds = Autoencoder_Conv_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_data = autoenc_conv_encode_data(cae, train_loader)
  
  return(encoded_data)


def autoenc_conv_encode_decode_data(cae, data_loader):
  # Encode the synthetic time series data using the trained cae
  encoded_decoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = cae.encoder(inputs)
      decoded = cae.decoder(encoded)
      encoded_decoded_data.append(decoded.detach().numpy())

  encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)

  return encoded_decoded_data


def autoenc_conv_encode_decode(cae, data, batch_size = 32):
  array = data.to_numpy()
  array = array[:, :, np.newaxis]
  
  ds = Autoencoder_Conv_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_decoded_data = autoenc_conv_encode_decode_data(cae, train_loader)
  
  return(encoded_decoded_data)
  
