from torch import nn
from typing import List
import torch
from torch.nn import functional as F



class PIA(nn.Module):

    def __init__(self,
                b_values = [0, 5, 50, 100, 200, 500, 800, 1000],
                hidden_dims= [32, 64, 128, 256, 512],
                predictor_depth=1):
        super(PIA, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.number_of_signals = len(b_values)
        self.b_values = b_values
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        modules = []
        # Build Encoder
        in_channels = number_of_signals
        for h_dim in hidden_dims:
            
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules).to(device)

        D_predictor = []
        for _ in range(predictor_depth):
            
            D_predictor.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1]),
                    nn.LeakyReLU())
            )
        D_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.D_predictor = nn.Sequential(*D_predictor).to(device)


        D_star_predictor = []
        for _ in range(predictor_depth):
            
            D_star_predictor.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1]),
                    nn.LeakyReLU())
            )
        D_star_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.D_star_predictor = nn.Sequential(*D_star_predictor).to(device)

        f_predictor = []
        for _ in range(predictor_depth):
            
            f_predictor.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1]),
                    nn.LeakyReLU())
            )
        f_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.f_predictor = nn.Sequential(*f_predictor).to(device)
        

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(x)
        
        D = 3*self.sigmoid(self.D_predictor(result))
        D_star = 3 + self.relu(self.D_star_predictor(result))
        f = self.sigmoid(self.f_predictor(result))
        
        return [f, D, D_star]

    def decode(self, f, D, D_star):
        """
            Maps the given latent codes onto the signal space.
            return: (Tensor) signal estimate
        """
        signal = torch.zeros((D.shape[0], self.number_of_signals))
        D, D_star, f = D.T, D_star.T, f.T
        for inx, b in enumerate(self.b_values):
            S = (1-f)*torch.exp(-b/1000*D) + f*torch.exp(-b/1000*D_star)
            signal[:, inx] = S
        return signal
    
    
    def forward(self, x):
        f, D, D_star = self.encode(x)
        return  [self.decode(f, D, D_star), x, f, D, D_star]
        # TODO: Add a control mechanism for unreasonable estimates


    def loss_function(self, recons, x):
        pred_signal = recons
        true_signal = x
        loss = F.mse_loss(pred_signal, true_signal)
        return loss
