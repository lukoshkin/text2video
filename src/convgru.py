import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN

class GeneralConvGRUCell(nn.Module):
    def __init__(self, conv_layers):
        super().__init__()
        self.ConvZ, self.ConvR, self.ConvH = conv_layers

    def forward(self, x, h_prev):
        XH = torch.cat([x, h_prev], dim=1)
        z = torch.sigmoid(self.ConvZ(XH))
        r = torch.sigmoid(self.ConvR(XH))
        XRH = torch.cat([x, r*h_prev], dim=1)
        h_new = torch.tanh(self.ConvH(XRH))

        return (1 - z) * h_prev + z * h_new



class ConvGRUCell(GeneralConvGRUCell):
    def __init__(
            self, in_channels, hidden_planes, 
            kernel_size, spectral_norm):
        padding = kernel_size // 2
        ConvZ = nn.Conv2d(
                in_channels+hidden_planes, 
                hidden_planes, kernel_size, 
                padding=padding)
        ConvR = nn.Conv2d(
                in_channels+hidden_planes, 
                hidden_planes, kernel_size, 
                padding=padding)
        ConvH = nn.Conv2d(
                in_channels+hidden_planes, 
                hidden_planes, kernel_size, 
                padding=padding)

        if spectral_norm:
            ConvZ = SN(ConvZ)
            ConvR = SN(ConvR)
            ConvH = SN(ConvH)
        else:
            nn.init.orthogonal_(ConvZ.weight)
            nn.init.orthogonal_(ConvR.weight)
            nn.init.orthogonal_(ConvH.weight)
            nn.init.constant_(ConvZ.bias, 0.)
            nn.init.constant_(ConvR.bias, 0.)
            nn.init.constant_(ConvH.bias, 0.)
        super().__init__((ConvZ, ConvR, ConvH))


class ConvGRU(nn.Module):
    """
    Args:
    seq_first       if seq_first is True, the layer
                    takes an input of the shape (N, T, C, H, W)
                    returns a tensor of the shape (N*T, K, H, W)
                    and the last hidden state of the size (N, K, H, W).
                    Otherwise, it expects the shape (N, C, T, H, W)
                    and outputs the shapes (N, K, T, H, W) and (N, K, H, W)
                    K - number of hidden planes 
                    C - number of the input channels
                    T - number of frames in a video
    """
    def __init__(
            self, in_channels, hidden_planes,
            kernel_size, seq_first=False,
            spectral_norm=False):
        super().__init__()
        self.C = hidden_planes
        self.seq_first = seq_first
        self.CGRUcell = ConvGRUCell(
                in_channels, hidden_planes,
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


class AdvancedConvGRU(nn.Module):
    """
    Args:
    deepConv2d      block of 2d convolutions of k.s. 3, stride 1, and pad. 1
    seq_first       if seq_first is True, the layer 
                    takes an input of the shape (N, T, C, H, W)
                    returns a tensor of the shape (N*T, K, H, W)
                    and the last hidden state of the size (N, K, H, W).
                    Otherwise, it expects the shape (N, C, T, H, W)
                    and outputs the shapes (N, K, T, H, W) and (N, K, H, W)
                    K - number of hidden planes
                    C - number of the input channels
                    T - number of frames in a video
    """
    def __init__(
            self, deepConv2d, in_channels, hidden_planes, seq_first=False):
        super().__init__()
        self._C = hidden_planes
        self.seq_first = seq_first
        in_channels += hidden_planes
        conv_layers = [deepConv2d(in_channels, hidden_planes)]
        conv_layers += [deepConv2d(in_channels, hidden_planes)]
        conv_layers += [deepConv2d(in_channels, hidden_planes)]
        self.CGRUcell = GeneralConvGRUCell(conv_layers)

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
            h_init = X.new(X.size(0), self._C, *X.shape[3:]).fill_(0)
        if self.seq_first:
            return self._seqFirstCall(X, h_init)
        else:
            return self._defaultCall(X, h_init)
