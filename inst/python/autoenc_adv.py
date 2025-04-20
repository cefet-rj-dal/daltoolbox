import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

torch.set_grad_enabled(True)


##################################
# Define Networks
##################################
# Encoder
class Q_net(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(input_size, 60)
        self.lin2 = nn.Linear(60, 60)
        # Gaussian code (z)
        self.lin3gauss = nn.Linear(60, encoding_size)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.4, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.4, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)

        return xgauss


# Decoder
class P_net(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(encoding_size, 60)
        self.lin2 = nn.Linear(60, 60)
        self.lin3 = nn.Linear(60, input_size)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)


class D_net_gauss(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(encoding_size, 60)
        self.lin2 = nn.Linear(60, 60)
        self.lin3 = nn.Linear(60, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.4, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.4, training=self.training)
        x = F.relu(x)

        return F.sigmoid(self.lin3(x))


def create_latent(Q, loader):
    '''
    Creates the latent representation for the samples in loader
    return:
        z_values: numpy array with the latent representations
        labels: the labels corresponding to the latent representations
    '''
    Q.eval()
    labels = []

    for batch_idx, (X, target) in enumerate(loader):

        X = X * 0.3081 + 0.1307
        # X.resize_(loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        labels.extend(target.data.tolist())
        if cuda:
            X, target = X.cuda(), target.cuda()
        # Reconstruction phase
        z_sample = Q(X)
        if batch_idx > 0:
            z_values = np.concatenate((z_values, np.array(z_sample.data.tolist())))
        else:
            z_values = np.array(z_sample.data.tolist())
    labels = np.array(labels)

    return z_values, labels


class Autoencoder_Adv_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.data[index]


class Autoencoder_Adv(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder_Adv, self).__init__()
        
        self.input_size = input_size
        self.encoding_size = encoding_size

        self.Q = Q_net(input_size, encoding_size)

        self.P = P_net(input_size, encoding_size)
        
        self.D_gauss = D_net_gauss(input_size, encoding_size)

        
# Create the aae
def autoenc_adv_create(input_size, encoding_size):
  input_size = int(input_size)
  encoding_size = int(encoding_size)
  
  torch.manual_seed(10)

  aae = Autoencoder_Adv(input_size, encoding_size)

  # Set learning rates
  gen_lr = 0.0001
  reg_lr = 0.00005

  # Set optimizators
  aae.encoder = optim.Adam(aae.Q.parameters(), lr=gen_lr)
  aae.decoder = optim.Adam(aae.P.parameters(), lr=gen_lr)

  aae.generator = optim.Adam(aae.Q.parameters(), lr=reg_lr)
  aae.D_gauss_solver = optim.Adam(aae.D_gauss.parameters(), lr=reg_lr)

  return aae  


# Train the aae
def autoenc_adv_train(aae, data, batch_size = 350, num_epochs = 1000, learning_rate = 0.001):
  recon_criterion = nn.MSELoss()
  
  TINY = 1e-15
  # Set the networks in train mode (apply dropout when needed)
  aae.Q.train()
  aae.P.train()
  aae.D_gauss.train()
  
  # Init gradients
  aae.P.zero_grad()
  aae.Q.zero_grad()
  aae.D_gauss.zero_grad()
  
  train_loss = []
  val_loss = []
  
  for epoch in range(num_epochs):
    array = data.to_numpy()
    array = array[:, :, np.newaxis]
    
    val_sample = sample(range(1, data.shape[0], 1), k=int(data.shape[0]*0.3))
    train_sample = [v for v in range(1, data.shape[0], 1) if v not in val_sample]
    
    train_data = array[train_sample, :, :]
    val_data = array[val_sample, :, :]
    
    ds_train = Autoencoder_Adv_TS(train_data)
    ds_val = Autoencoder_Adv_TS(val_data)
    
    train_loader = DataLoader(ds_train, batch_size=batch_size)
    val_loader = DataLoader(ds_val, batch_size=batch_size)
    
    #Train
    train_epoch_loss = []
    val_epoch_loss = []
    
    for train_data in train_loader:
        train_input, _ = train_data
        train_input = train_input.float()
        train_input = train_input.view(train_input.size(0), -1)

        #######################
        # Reconstruction phase
        #######################
        z_sample = aae.Q(train_input)
        X_sample = aae.P(z_sample)
        recon_loss = recon_criterion(X_sample + TINY, train_input + TINY)

        recon_loss.backward()
        aae.decoder.step()
        aae.encoder.step()

        aae.P.zero_grad()
        aae.Q.zero_grad()
        aae.D_gauss.zero_grad()

        #######################
        # Regularization phase
        #######################
        # Discriminator
        aae.Q.eval()
        z_real_gauss = Variable(torch.randn(len(train_input), aae.encoding_size) * 5.)

        z_fake_gauss = aae.Q(train_input)

        D_real_gauss = aae.D_gauss(z_real_gauss)
        D_fake_gauss = aae.D_gauss(z_fake_gauss)

        D_loss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

        D_loss.backward()
        aae.D_gauss_solver.step()

        aae.P.zero_grad()
        aae.Q.zero_grad()
        aae.D_gauss.zero_grad()

        # Generator
        aae.Q.train()
        z_fake_gauss = aae.Q(train_input)

        D_fake_gauss = aae.D_gauss(z_fake_gauss)
        G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))

        G_loss.backward()
        aae.generator.step()

        aae.P.zero_grad()
        aae.Q.zero_grad()
        aae.D_gauss.zero_grad()
        train_epoch_loss.append(recon_loss.item())
        
    for val_data in val_loader:
        val_input, _ = val_data
        val_input = val_input.float()
        val_input = val_input.view(val_input.size(0), -1)
        
        val_z = aae.Q(val_input)
        val_output = aae.P(val_z)
        
        val_batch_loss = recon_criterion(val_output + TINY, val_input + TINY)
        val_epoch_loss.append(val_batch_loss.item())
        train_epoch_loss.append(val_batch_loss.item())
    
        
    train_loss.append(np.mean(train_epoch_loss))
    val_loss.append(np.mean(val_epoch_loss))
    
  return aae, np.array(train_loss), np.array(val_loss)



def autoenc_adv_fit(aae, data, batch_size = 350, num_epochs = 1000, learning_rate = 0.001):
  batch_size = int(batch_size)
  num_epochs = int(num_epochs)
  
  aae = autoenc_adv_train(aae, data, batch_size = batch_size, num_epochs=num_epochs, learning_rate=learning_rate)
  
  return aae


def autoenc_adv_encode_data(aae, data_loader):
  # Encode the synthetic time series data using the trained aae
  encoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      inputs = inputs.view(inputs.size(0), -1)
      encoded = aae.Q(inputs)
      encoded_data.append(encoded.detach().numpy())

  encoded_data = np.concatenate(encoded_data, axis=0)

  return encoded_data

def autoenc_adv_encode(aae, data, batch_size = 32):
  array = data.to_numpy()
  array = array[:, :, np.newaxis]
  
  ds = Autoencoder_Adv_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_data = autoenc_adv_encode_data(aae, train_loader)
  
  return(encoded_data)


def autoenc_adv_encode_decode_data(aae, data_loader):
  aae.Q.eval()
  aae.P.eval()
  
  # Encode the synthetic time series data using the trained aae
  encoded_decoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      inputs = inputs.view(inputs.size(0), -1)
      
      z = aae.Q(inputs)
      decoded = aae.P(z)
      
      encoded_decoded_data.append(decoded.detach().numpy())

  encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)

  return encoded_decoded_data


def autoenc_adv_encode_decode(aae, data, batch_size = 350):
  array = data.to_numpy()
  array = array[:, :, np.newaxis]
  
  ds = Autoencoder_Adv_TS(array)
  pred_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_decoded_data = autoenc_adv_encode_decode_data(aae, pred_loader)
  
  return(encoded_decoded_data)
  
