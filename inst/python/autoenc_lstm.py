import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.set_grad_enabled(True)
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Encoder(nn.Module):
    def __init__(self, input_size, encoding_size, n_layers=1, dropout=0):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(
          input_size=input_size,
          hidden_size=encoding_size,
          dropout=dropout,
          num_layers=n_layers,
          batch_first=True  # True = (batch_size, seq_len, input_size)
                            # False = (seq_len, batch_size, input_size) 
                            #default = false
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1].unsqueeze(1)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, encoding_size, n_layers=1, dropout=0):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(
        input_size=encoding_size,
        hidden_size=encoding_size,
        dropout=dropout,
        num_layers=n_layers,
        batch_first=True
        )
        
        self.output_layer = nn.Linear(encoding_size, input_size)

    def forward(self, x):
        x=self.lstm(x)
        x=x[0]
        x=self.output_layer(x)
        x=x.reshape((x.shape[0], x.shape[2], x.shape[1]))
        return x

class Autoencoder_LSTM_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.data[index]

class LSTM(nn.Module):
    def __init__(self, input_size, encoding_size, n_layers=1, dropout=0):
        super(LSTM, self).__init__()

        self.encoder = Encoder(input_size, encoding_size, n_layers, dropout)
        self.decoder = Decoder(input_size, encoding_size, n_layers, dropout)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# Create the autoencoder
def autoenc_lstm_create(input_size, encoding_size):
  input_size = int(input_size)
  encoding_size = int(encoding_size)
  
  lae = LSTM(input_size, encoding_size)
  lae = lae.float()
  return lae  

# Train the lae
def autoenc_lstm_train(lae, data, batch_size=20, num_epochs = 1000, learning_rate = 0.00001):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(lae.parameters(), lr=learning_rate)

  train_loss = []
  val_loss = []
  
  for epoch in range(num_epochs):
      # Train Test Split
      array = data.to_numpy()
      array = array.reshape(array.shape[0], 1, array.shape[1])
      
      val_sample = sample(range(1, data.shape[0], 1), k=int(data.shape[0]*0.3))
      train_sample = [v for v in range(1, data.shape[0], 1) if v not in val_sample]
      
      train_data = array[train_sample, :]
      val_data = array[val_sample, :]
      
      ds_train = Autoencoder_LSTM_TS(train_data)
      ds_val = Autoencoder_LSTM_TS(val_data)
      train_loader = DataLoader(ds_train, batch_size=batch_size)
      val_loader = DataLoader(ds_val, batch_size=batch_size)
    
      # Train
      train_epoch_loss = []
      val_epoch_loss = []
      lae.train()
      for train_data in train_loader:
          train_input, _ = train_data
          train_input = train_input.float()
          optimizer.zero_grad()
          train_output = lae(train_input)
          train_batch_loss = criterion(train_output, train_input)
          train_batch_loss.backward()
          optimizer.step()
          train_epoch_loss.append(train_batch_loss.item())
          
          
      # Validation
      lae.eval()
      for val_data in val_loader:
          val_input, _ = val_data
          val_input = val_input.float()
          val_output = lae(val_input)
          val_batch_loss = criterion(val_output, val_input)
          val_epoch_loss.append(val_batch_loss.item())
          
      train_loss.append(np.mean(train_epoch_loss))
      val_loss.append(np.mean(val_epoch_loss))
  
  return lae, np.array(train_loss), np.array(val_loss)  


def autoenc_lstm_fit(lae, data, batch_size = 20, num_epochs = 1000, learning_rate = 0.001, return_loss=False):
  batch_size = int(batch_size)
  num_epochs = int(num_epochs)

  lae = autoenc_lstm_train(lae, data, batch_size = batch_size, num_epochs = num_epochs, learning_rate = learning_rate)
  return lae


def autoenc_lstm_encode_data(lae, data_loader):
  # Encode the synthetic time series data using the trained autoencoder
  encoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = lae.encoder(inputs)
      encoded_data.append(encoded.detach().numpy())

  encoded_data = np.concatenate(encoded_data, axis=0)

  return encoded_data

def autoenc_lstm_encode(lae, data, batch_size = 20):
  array = data.to_numpy()
  array = array.reshape(array.shape[0], 1, array.shape[1])
  
  ds = Autoencoder_LSTM_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_data = autoenc_lstm_encode_data(lae, train_loader)
  
  encoded_data = encoded_data.reshape((encoded_data.shape[0], encoded_data.shape[2]))
  
  return(encoded_data)


def autoenc_lstm_encode_decode_data(lae, data_loader):
  # Encode the synthetic time series data using the trained autoencoder
  encoded_decoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = lae.encoder(inputs)
      decoded = lae.decoder(encoded)
      encoded_decoded_data.append(decoded.detach().numpy())

  encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)

  return encoded_decoded_data


def autoenc_lstm_encode_decode(lae, data, batch_size = 20):
  array = data.to_numpy()
  array = array.reshape(array.shape[0], 1, array.shape[1])
  
  ds = Autoencoder_LSTM_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_decoded_data = autoenc_lstm_encode_decode_data(lae, train_loader)
  
  return(encoded_decoded_data)
  
