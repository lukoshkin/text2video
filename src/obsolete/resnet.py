import torch
from torch import nn


class Noise(nn.Module):
    def __init__(self, noise, sigma=.1):
        super().__init__()
        self.noise = noise
        self.sigma = sigma

    def forward(self, x):
        if self.noise:
            return x + self.sigma * torch.randn_like(x)
        return x


def block1x1(Conv, BatchNorm, inC, outC, noise, sigma):
    """
    Returns a layer which performs a 1x1 convolution
    (1x1x1 in case of 3d) with subsequent normalization 
    and rectification
    """
    block_list = [Noise(noise, sigma)]
    block_list += [Conv(inC, outC, 1, bias=False)]
    if BatchNorm is not None:
        block_list += [BatchNorm(outC)]

    return nn.Sequential(*block_list)


def block3x3(Conv, BatchNorm, inC, outC, stride, noise, sigma):
    """
    Returns a layer which performs a 3x3 convolution
    (3x3x3 in case of 3d) with subsequent normalization 
    and rectification
    """
    block_list = [Noise(noise, sigma)]
    block_list += [Conv(inC, outC, 3, stride, 1, bias=False)]
    if BatchNorm is not None:
        block_list += [BatchNorm(outC)]

    return nn.Sequential(*block_list)


class BasicBlock(nn.Module):
    """
    Args:
        type        2d or 3d
        stride      convolution stride (int or tuple of ints)
        noise       boolen flag: use Noise layer or do not
        sigma       standard deviation of the gaussian noise
                    used in Noise layer
        bn          whether to add BatchNorm layer
    """
    def __init__(
            self, type, in_channels, out_channels, 
            stride=1, bn=True, noise=False, sigma=.2):
        super().__init__()

        if type == '3d': 
            Conv = nn.Conv3d
            AvgPool = nn.AvgPool3d
            BatchNorm = nn.BatchNorm3d if bn else None
        elif type == '2d':
            Conv = nn.Conv2d
            AvgPool = nn.AvgPool2d
            BatchNorm = nn.BatchNorm2d if bn else None
        else:
            raise TypeError (
                "__init__(): argument 'type' "
                "must be '2d' or '3d'"
            )
        proj_list = []
        self.proj = None
        assert (torch.tensor(stride) <= 2).all() 
        if in_channels != out_channels:
            proj_list += [Conv(in_channels, out_channels, 1)]
        if (torch.tensor(stride) > 1).any():
            proj_list += [AvgPool(stride)]
        self.proj = nn.Sequential(*proj_list)

        self.leaky = nn.LeakyReLU(.2, inplace=True)
        self.main = nn.Sequential (
            block3x3(
                Conv, BatchNorm, in_channels, 
                out_channels, stride, noise, sigma),
            self.leaky,
            block3x3(
                Conv, BatchNorm, out_channels, 
                out_channels, 1, noise, sigma),
        )

    def forward(self, x):
        y = self.main(x)
        if self.proj is not None:
            x = self.proj(x)
            
        return self.leaky(y + x)


class Bottleneck(nn.Module):
    """
    Args:
        type        2d or 3d
        width       width of bottleneck
        stride      convolution stride (int or tuple of ints)
        noise       boolen flag: use Noise layer or do not
        sigma       standard deviation of the gaussian noise
                    used in Noise layer
        bn          whether to add BatchNorm layer
    """
    def __init__(
            self, type, in_channels, out_channels, stride=1, 
            bn=True, width=None, noise=False, sigma=.2):
        super().__init__()

        if type == '3d': 
            Conv = nn.Conv3d
            AvgPool = nn.AvgPool3d
            BatchNorm = nn.BatchNorm3d if bn else None
        elif type == '2d':
            Conv = nn.Conv2d
            AvgPool = nn.AvgPool2d
            BatchNorm = nn.BatchNorm2d if bn else None
        else:
            raise TypeError (
                "__init__(): argument 'type' "
                "must be '2d' or '3d'"
            )
        proj_list = []
        self.proj = None
        assert (torch.tensor(stride) <= 2).all() 
        if in_channels != out_channels:
            proj_list += [Conv(in_channels, out_channels, 1)]
        if (torch.tensor(stride) > 1).any():
            proj_list += [AvgPool(stride)]
        self.proj = nn.Sequential(*proj_list)

        if not width:
            width = (in_channels + out_channels) // 4

        self.leaky = nn.LeakyReLU(.2, inplace=True)
        self.main = nn.Sequential (
            block1x1(Conv, BatchNorm, in_channels, width, noise, sigma),
            self.leaky,
            block3x3(Conv, BatchNorm, width, width, stride, noise, sigma),
            self.leaky,
            block1x1(Conv, BatchNorm, width, out_channels, noise, sigma)
        )

    def forward(self, x):
        y = self.main(x)
        if self.proj is not None:
            x = self.proj(x)
            
        return self.leaky(y + x)
