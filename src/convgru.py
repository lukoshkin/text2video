import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN


class ConvGRUCell(nn.Module):
    def __init__(
            self, in_channels, hidden_layers, 
            kernel_size, spectral_norm):
        super().__init__()
        padding = kernel_size // 2
        self.ConvZ = nn.Conv2d(
                in_channels+hidden_layers, 
                hidden_layers, kernel_size, 
                padding=padding)
        self.ConvR = nn.Conv2d(
                in_channels+hidden_layers, 
                hidden_layers, kernel_size, 
                padding=padding)
        self.ConvH = nn.Conv2d(
                in_channels+hidden_layers, 
                hidden_layers, kernel_size, 
                padding=padding)

        if spectral_norm:
            self.ConvZ = SN(self.ConvZ)
            self.ConvR = SN(self.ConvR)
            self.ConvH = SN(self.ConvH)
        else:
            nn.init.orthogonal_(self.ConvZ.weight)
            nn.init.orthogonal_(self.ConvR.weight)
            nn.init.orthogonal_(self.ConvH.weight)
            nn.init.constant_(self.ConvZ.bias, 0.)
            nn.init.constant_(self.ConvR.bias, 0.)
            nn.init.constant_(self.ConvH.bias, 0.)


    def forward(self, x, h_prev):
        XH = torch.cat([x, h_prev], dim=1)
        z = torch.sigmoid(self.ConvZ(XH))
        r = torch.sigmoid(self.ConvR(XH))
        XRH = torch.cat([x, r*h_prev], dim=1)
        h_new = torch.tanh(self.ConvH(XRH))

        return (1 - z) * h_prev + z * h_new 


class ConvGRU(nn.Module):
    """
    Args:
    seq_first       if seq_first is True, the layer 
                    takes an input of the shape (N, T, C, H, W)
                    returns a tensor of the shape (N*T, K, H, W)
                    and the last hidden state of the size (N, K, H, W).
                    Otherwise, it expects the shape (N, C, T, H, W)
                    and outputs the shapes (N, K, T, H, W) and (N, K, H, W)
                    K - number of hidden layers
                    C - number of the input channels
                    T - number of frames in a video
    """
    def __init__(
            self, in_channels, hidden_layers, 
            kernel_size, seq_first=False, 
            spectral_norm=False):
        super().__init__()
        self.C = hidden_layers
        self.seq_first = seq_first
        self.CGRUcell = ConvGRUCell(
                in_channels, hidden_layers, 
                kernel_size, spectral_norm)

    def _defaultCall(self, X, h_init):
        H = [h_init]
        for i in range(X.size(2)):
            H.append(self.CGRUcell(X[:, :, i], H[-1]))
        return torch.stack(H[1:], dim=2), H[-1]

    def _seqFirstCall(self, X, h_init):
        H = [h_init]
        for i in range(X.size(1)):
            H.append(self.CGRUcell(X[:, i], H[-1]))
        return torch.flatten(torch.stack(H[1:], dim=1), 0, 1), H[-1]
    
    def forward(self, X, h_init=None):
        if h_init is None:
            h_init = X.new(X.size(0), self.C, *X.shape[3:]).fill_(0)
        if self.seq_first:
            return self._seqFirstCall(X, h_init)
        else:
            return self._defaultCall(X, h_init)
