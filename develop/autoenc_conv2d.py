import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Autoencoder_Conv2D_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index-1], self.data[index-1]

class Autoencoder_Conv2D(nn.Module):
    def __init__(self, input_size, encoding_size, filter_size=1, kernel_size=5, padding=1, stride=4, dilation=1):
        super(Autoencoder_Conv2D, self).__init__()
        
        output_size = int((76+(2*padding) - dilation * (kernel_size - 1) - 1)/stride) + 1
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * output_size * output_size, encoding_size),
            nn.ReLU(True)
            )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 16 * output_size * output_size),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(16, output_size, output_size)),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 6, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(True)
            )
    
    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x

    
# Create the cae
def autoenc_conv2d_create(input_size, encoding_size):
  input_size = tuple(input_size)
  encoding_size = int(encoding_size)
  
  cae2d = Autoencoder_Conv2D(input_size, encoding_size)
  cae2d = cae2d.float()
  return cae2d  

# Train the cae
def autoenc_conv2d_train(cae2d, data, batch_size=20, num_epochs = 1000, learning_rate = 0.001):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(cae2d.parameters(), lr=learning_rate)

  train_loss = []
  val_loss = []
  
  for epoch in range(num_epochs):
      # Train Test Split
      val_sample = sample(range(1, data.shape[0], 1), k=int(data.shape[0]*0.3))
      train_sample = [v for v in range(1, data.shape[0], 1) if v not in val_sample]
      
      train_data = data[train_sample, :, :, :]
      val_data = data[val_sample, :, :, :]
      
      ds_train = Autoencoder_Conv2D_TS(train_data)
      ds_val = Autoencoder_Conv2D_TS(val_data)
      train_loader = DataLoader(ds_train, batch_size=batch_size)
      val_loader = DataLoader(ds_val, batch_size=batch_size)
    
      # Train
      train_epoch_loss = []
      val_epoch_loss = []
      cae2d.train()
      for train_data in train_loader:
          train_input, _ = train_data
          train_input = train_input.float()
          optimizer.zero_grad()
          train_output = cae2d(train_input)
          train_batch_loss = criterion(train_output, train_input)
          train_batch_loss.backward()
          optimizer.step()
          train_epoch_loss.append(train_batch_loss.item())
          
          
      # Validation
      cae2d.eval()
      for val_data in val_loader:
          val_input, _ = val_data
          val_input = val_input.float()
          val_output = cae2d(val_input)
          val_batch_loss = criterion(val_output, val_input)
          val_epoch_loss.append(val_batch_loss.item())
          
      train_loss.append(np.mean(train_epoch_loss))
      val_loss.append(np.mean(val_epoch_loss))
  
  cae2d.train_loss = train_loss
  cae2d.val_loss = val_loss
  return cae2d

def autoenc_conv2d_fit(cae2d, data, batch_size = 20, num_epochs = 50, learning_rate = 0.001):
  batch_size = int(batch_size)
  num_epochs = int(num_epochs)

  cae2d = autoenc_conv2d_train(cae2d, data, batch_size=batch_size, num_epochs = num_epochs, learning_rate = learning_rate)
  return cae2d

def autoenc_conv2d_encode_data(cae2d, data_loader):
  # Encode the image data using the trained cae2d
  encoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = cae2d.encoder(inputs)
      encoded_data.append(encoded.detach().numpy())

  encoded_data = np.concatenate(encoded_data, axis=0)

  return encoded_data

def autoenc_conv2d_encode(cae2d, data, batch_size = 32):
  ds = Autoencoder_Conv2D_TS(data)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_data = autoenc_conv2d_encode_data(cae2d, train_loader)
  
  return(encoded_data)


def autoenc_conv2d_encode_decode_data(cae2d, data_loader):
  # Encode the image data using the trained cae2d
  encoded_decoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = cae2d.encoder(inputs)
      decoded = cae2d.decoder(encoded)
      encoded_decoded_data.append(decoded.detach().numpy())

  encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)

  return encoded_decoded_data


def autoenc_conv2d_encode_decode(cae2d, data, batch_size = 32):
  ds = Autoencoder_Conv2D_TS(data)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_decoded_data = autoenc_conv2d_encode_decode_data(cae2d, train_loader)
  
  return(encoded_decoded_data)
  
