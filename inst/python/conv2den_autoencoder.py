import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import sample


class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim, padding=0):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 8, 3, stride=2, padding=padding),
            #nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Third convolutional layer
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            #nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(148 * 296, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = self.flatten(x)
        # # Apply linear layers
        x = self.encoder_lin(x)
        return x
  
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim, fc2_input_dim, padding=0):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, 148 * 296),
            nn.ReLU(True)
        )

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 37, 37))

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=padding)
        )
        
    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x
  
class C2DEN_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index-1], self.data[index-1]

class C2DEN(nn.Module):
    def __init__(self, input_size, encoding_size, filter_size=1, kernel_size=4, padding=0, stride=1):
        super(C2DEN, self).__init__()
        
        self.encoder = Encoder(encoded_space_dim=encoding_size, padding=padding, fc2_input_dim=128)
        self.decoder = Decoder(encoded_space_dim=encoding_size, padding=padding, fc2_input_dim=128)
        
    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x

# Create the cae
def c2den_create(input_size, encoding_size):
  input_size = tuple(input_size)
  encoding_size = int(encoding_size)
  
  c2den = C2DEN(input_size, encoding_size)
  c2den = c2den.float()
  return c2den  

# Train the cae
def c2den_train(c2den, data, batch_size=20, num_epochs = 1000, learning_rate = 0.001):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(c2den.parameters(), lr=learning_rate)

  train_loss = []
  val_loss = []
  
  for epoch in range(num_epochs):
      # Train Test Split
      val_sample = sample(range(1, data.shape[0], 1), k=int(data.shape[0]*0.3))
      train_sample = [v for v in range(1, data.shape[0], 1) if v not in val_sample]
      
      train_data = data[train_sample, :, :, :]
      val_data = data[val_sample, :, :, :]
      
      ds_train = C2DEN_TS(train_data)
      ds_val = C2DEN_TS(val_data)
      train_loader = DataLoader(ds_train, batch_size=batch_size)
      val_loader = DataLoader(ds_val, batch_size=batch_size)
    
      # Train
      train_epoch_loss = []
      val_epoch_loss = []
      c2den.train()
      for train_data in train_loader:
          train_input, _ = train_data
          train_input = train_input.float()
          optimizer.zero_grad()
          train_output = c2den(train_input)
          train_batch_loss = criterion(train_output, train_input)
          train_batch_loss.backward()
          optimizer.step()
          train_epoch_loss.append(train_batch_loss.item())
          
          
      # Validation
      c2den.eval()
      for val_data in val_loader:
          val_input, _ = val_data
          val_input = val_input.float()
          val_output = c2den(val_input)
          val_batch_loss = criterion(val_output, val_input)
          val_epoch_loss.append(val_batch_loss.item())
          
      train_loss.append(np.mean(train_epoch_loss))
      val_loss.append(np.mean(val_epoch_loss))
  
  c2den.train_loss = train_loss
  c2den.val_loss = val_loss
  return c2den

def c2den_fit(c2den, data, batch_size = 20, num_epochs = 50, learning_rate = 0.001):
  batch_size = int(batch_size)
  num_epochs = int(num_epochs)

  c2den = c2den_train(c2den, data, batch_size=batch_size, num_epochs = num_epochs, learning_rate = learning_rate)
  return c2den

def c2den_encode_data(c2den, data_loader):
  # Encode the synthetic time series data using the trained cae
  encoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = c2den.encoder(inputs)
      encoded_data.append(encoded.detach().numpy())

  encoded_data = np.concatenate(encoded_data, axis=0)

  return encoded_data

def c2den_encode(c2den, data, batch_size = 32):

  ds = C2DEN_TS(data)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_data = c2den_encode_data(c2den, train_loader)
  
  return(encoded_data)


def c2den_encode_decode_data(c2den, data_loader):
  # Encode the synthetic time series data using the trained cae
  encoded_decoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = c2den.encoder(inputs)
      decoded = c2den.decoder(encoded)
      encoded_decoded_data.append(decoded.detach().numpy())

  encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)

  return encoded_decoded_data


def c2den_encode_decode(c2den, data, batch_size = 32):
  ds = C2DEN_TS(data)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_decoded_data = c2den_encode_decode_data(c2den, train_loader)
  
  return(encoded_decoded_data)
  
