import torch
from torch import nn
from torch.nn.utils import spectral_norm

class DBlock(nn.Module):
    """
    Discriminator's block
    """
    def __init__(
            self, type, 
            in_channels, out_channels, stride, 
            bn=False, sn=True):
        super().__init__()
        if type == '2d':
            Conv = nn.Conv2d
            AvgPool = nn.AvgPool2d
            BatchNorm = nn.BatchNorm2d
        elif type == '3d':
            Conv = nn.Conv3d
            AvgPool = nn.AvgPool3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise TypeError (
                "__init__(): argument 'type' "
                "must be '2d' or '3d'"
            )
        SN = spectral_norm if sn else lambda x: x
        BatchNorm = BatchNorm if bn else lambda x: nn.Sequential()
            
        main_list = [
            BatchNorm(in_channels),
            nn.LeakyReLU(.2, inplace=True),
            SN(Conv(in_channels, out_channels, 3, 1, 1)),
            BatchNorm(out_channels),
            nn.LeakyReLU(.2, inplace=True),
            SN(Conv(out_channels, out_channels, 3, 1, 1))
        ]
        proj_list = []
        self.proj = None
        assert (torch.tensor(stride) <= 2).all()
        if in_channels != out_channels:
            # in some models spectral norm reduces 
            # the stability when applied to shortcut
            proj_list += [SN(Conv(in_channels, out_channels, 1))]
        if (torch.tensor(stride) > 1).any():
            main_list += [AvgPool(stride)]
            proj_list += [AvgPool(stride)]
        self.proj = nn.Sequential(*proj_list)
        self.main = nn.Sequential(*main_list)

    def forward(self, x):
        return self.main(x) + self.proj(x)


#global variable for the rest of the classes
SN = spectral_norm


class GBlock(nn.Module):
    """
    Generator's block
    """
    def __init__(self, type, in_channels, out_channels, stride):
        super().__init__()
        if type == '2d':
            Conv = nn.Conv2d
            AvgPool = nn.AvgPool2d
            BatchNorm = nn.BatchNorm2d
        elif type == '3d':
            Conv = nn.Conv3d
            AvgPool = nn.AvgPool3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise TypeError (
                "__init__(): argument 'type' "
                "must be '2d' or '3d'"
            )
        main_list = [
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            SN(Conv(in_channels, out_channels, 3, 1, 1)),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            SN(Conv(out_channels, out_channels, 3, 1, 1)),
        ]
        proj_list = []
        self.proj = None
        assert (torch.tensor(stride) <= 2).all()
        if in_channels != out_channels:
            # in some models spectral norm reduces 
            # the stability when applied to shortcut
            proj_list += [SN(Conv(in_channels, out_channels, 1))]
        if (torch.tensor(stride) > 1).any():
            main_list.insert(2, nn.Upsample(scale_factor=stride))
            proj_list.insert(0, nn.Upsample(scale_factor=stride))
        self.proj = nn.Sequential(*proj_list)
        self.main = nn.Sequential(*main_list)

    def forward(self, x):
        return self.main(x) + self.proj(x)


class CGBlock(nn.Module):
    """
    Conditional generator's block
    """
    def __init__(self, type, cond_size, in_channels, out_channels, stride):
        super().__init__()
        if type == '2d':
            Conv = nn.Conv2d
            AvgPool = nn.AvgPool2d
        elif type == '3d':
            Conv = nn.Conv3d
            AvgPool = nn.AvgPool3d
        else:
            raise TypeError (
                "__init__(): argument 'type' "
                "must be '2d' or '3d'"
            )
        self.bn1 = CBN(type, cond_size, in_channels)
        self.bn2 = CBN(type, cond_size, out_channels)
        self.conv1 = SN(Conv(in_channels, out_channels, 3, 1, 1))
        self.conv2 = SN(Conv(out_channels, out_channels, 3, 1, 1))
        self.upsample = nn.Upsample(scale_factor=stride)
        self.relu = nn.ReLU(inplace=True)

        proj_list = []
        self.proj = None
        assert (torch.tensor(stride) <= 2).all()
        if in_channels != out_channels:
            # in some models spectral norm reduces 
            # the stability when applied to shortcut
            proj_list += [SN(Conv(in_channels, out_channels, 1))]
        if (torch.tensor(stride) > 1).any():
            proj_list.insert(0, nn.Upsample(scale_factor=stride))
        self.proj = nn.Sequential(*proj_list)

    def forward(self, x, y):
        h = self.relu(self.bn1(x, y))
        h = self.upsample(h)
        h = self.conv1(h)
        h = self.relu(self.bn2(h, y))
        h = self.conv2(h)

        return h + self.proj(x)


class CBN(nn.Module):
    """
    Conditional BatchNorm
    """
    def __init__(
            self, type, cond_size, out_channels, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gain = SN(nn.Linear(cond_size, out_channels, bias=False))
        self.bias = SN(nn.Linear(cond_size, out_channels, bias=False))
        if type == '2d':
            self.type = 2 
            self.bn = nn.BatchNorm2d(
                out_channels, eps, momentum, affine=False)
        else:
            self.type = 3
            self.bn = nn.BatchNorm3d(
                out_channels, eps, momentum, affine=False)

    def forward(self, x, y):
        gain = self.gain(y).view(y.size(0), -1, *(self.type*[1]))
        bias = self.bias(y).view(y.size(0), -1, *(self.type*[1]))
        
        return gain * self.bn(x) + bias
