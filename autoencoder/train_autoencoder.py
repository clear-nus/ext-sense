import numpy as np

import torch
import torch.nn as nn

from skorch import NeuralNetRegressor
from skorch.dataset import CVSplit
from skorch.callbacks import EarlyStopping, Checkpoint


def _generate_tool_autoencoder(signal_type, device):
    
    default_frequency = 4000 if signal_type == 'nuskin' else 2200
    
    npzfile20 = np.load(f'../data/preprocessed/{signal_type}_tool_20_{default_frequency}hz.npz')
    npzfile30 = np.load(f'../data/preprocessed/{signal_type}_tool_30_{default_frequency}hz.npz')
    npzfile50 = np.load(f'../data/preprocessed/{signal_type}_tool_50_{default_frequency}hz.npz')
    
    scale_factor = 40 if signal_type == 'nuskin' else 1000
    
    X20 = npzfile20['signals'] / scale_factor
    X30 = npzfile30['signals'] / scale_factor
    X50 = npzfile50['signals'] / scale_factor
    
    X20 = torch.Tensor(np.reshape(X20, (X20.shape[0], -1)))
    X30 = torch.Tensor(np.reshape(X30, (X30.shape[0], -1)))
    X50 = torch.Tensor(np.reshape(X50, (X50.shape[0], -1)))
    
    X = np.vstack((X20, X30, X50))
    
    callbacks = [
        EarlyStopping(patience=100),
        Checkpoint(dirname=f'models/{signal_type}_tool_autoencoder')
    ]
    
    estimator = NeuralNetRegressor(Autoencoder,
                                   module__input_dim=X.shape[1],
                                   module__latent_dim=32,
                                   optimizer=torch.optim.Adam,
                                   train_split=CVSplit(10),
                                   max_epochs=300,
                                   device='cuda:0',
                                   lr=0.001,
                                   verbose=1,
                                   callbacks=callbacks)
    
    estimator.fit(X, X)
    
    latent20 = estimator.module_.encode(X20.to(device))
    latent30 = estimator.module_.encode(X30.to(device))
    latent50 = estimator.module_.encode(X50.to(device))
    
    np.save(f'{signal_type}_tool_20_latent.npy', latent20.detach().cpu().numpy())
    np.save(f'{signal_type}_tool_30_latent.npy', latent30.detach().cpu().numpy())
    np.save(f'{signal_type}_tool_50_latent.npy', latent50.detach().cpu().numpy())


class Autoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        
        super(Autoencoder, self).__init__()
        
        self.encoder1 = nn.Linear(input_dim, 128)
        self.encoder2 = nn.Linear(128, 64)
        self.encoder3 = nn.Linear(64, latent_dim)
        
        self.decoder1 = nn.Linear(latent_dim, 64)
        self.decoder2 = nn.Linear(64, 128)
        self.decoder3 = nn.Linear(128, input_dim)

    def forward(self, X):
        
        latent = self.encode(X)
        X_reconstruct = self.decode(latent)
        
        return X_reconstruct
    
    def encode(self, X):
        
        latent = torch.relu(self.encoder1(X))
        latent = torch.relu(self.encoder2(latent))
        latent = self.encoder3(latent)
        
        return latent
    
    def decode(self, latent):
        
        X_reconstruct = torch.relu(self.decoder1(latent))
        X_reconstruct = torch.relu(self.decoder2(X_reconstruct))
        X_reconstruct = self.decoder3(X_reconstruct)
        
        return X_reconstruct


if __name__ == '__main__':
    
    _generate_tool_autoencoder('biotac', 'cuda:0')
    _generate_tool_autoencoder('nuskin', 'cuda:0')
