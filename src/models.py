import torch
import torch.nn as nn
import torch.utils.data

from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence 

class Noise(nn.Module):
    def __init__(self, noise, sigma=0.2):
        super().__init__()
        self.noise = noise
        self.sigma = sigma

    def forward(self, x):
        if self.noise:
            return x + self.sigma * torch.randn_like(x)
        return x


class TextEncoder(nn.Module):
    """
    Args:
        n_spots         number of vectors in attention matrix
        emb_weights     matrix of size (n_tokens, emb_dim) 

        set of the rest ('extra') tuning hyperparameters

        hyppar[0]       number of gru hidden units
        hyppar[1]       projection dimensionality
    """
    def __init__(
            self, emb_weights, 
            batch_size=None, hyppar=(64,64)):
        super().__init__()
        self.initial = None
        self.embed = nn.Embedding.from_pretrained (
                        emb_weights, 
                        freeze=False,
                        padding_idx=0 
                    )
        self.gru = nn.GRU(
            emb_weights.size(1), hyppar[0], 
            batch_first=True, bidirectional=True
        )
        self.proj = nn.Sequential (
            nn.Linear(hyppar[0]*2, hyppar[1]),
            nn.LeakyReLU(.2, inplace=True)
        )
        if batch_size:
            self.initial = (2, batch_size, hyppar[0])

    def forward(self, text_ids, lengths):
        lengths, sortbylen = lengths.sort(0, descending=True)
        H = self.embed(text_ids[sortbylen])
        H = pack_padded_sequence(H, lengths, batch_first=True)

        if self.initial is not None:
            H, last = self.gru(H, self.initial)
        else:
            H, last = self.gru(H)

        return H, last
        cuts = torch.cumsum(lengths-1, 0)

        return self.proj(H[0][cuts, :])


def block1x1(Conv, inC, outC, noise, sigma, bn):
    """
    Performs a 1x1 convolution (1x1x1 in case of 3d)
    with subsequent normalization and rectification
    """
    if Conv == nn.Conv3d: 
        BatchNorm = nn.BatchNorm3d
    elif Conv == nn.Conv2d: 
        BatchNorm = nn.BatchNorm2d
    else:
        raise TypeError (
            "__init__(): argument 'Conv' "
            "must be 'nn.Conv2d' or 'nn.Conv3d' instance"
        )
    block_list = [Noise(noise, sigma)]
    block_list += [Conv(inC, outC, 1, bias=False)]
    if bn:
        block_list += [BatchNorm(outC)]
    block_list += [nn.LeakyReLU(.2, inplace=True)]

    return nn.Sequential(*block_list)


def block3x3(Conv, width, stride, noise, sigma=.1, bn=True):
    """
    Performs a 3x3 convolution (3x3x3 in case of 3d)
    with subsequent normalization and rectification
    """
    if Conv == nn.Conv3d: 
        BatchNorm = nn.BatchNorm3d
    elif Conv == nn.Conv2d: 
        BatchNorm = nn.BatchNorm2d
    else:
        raise TypeError (
            "__init__(): argument 'Conv' "
            "must be 'nn.Conv2d' or 'nn.Conv3d' instance"
        )
    block_list = [Noise(noise, sigma)]
    block_list += [Conv(width, width, 3, stride, 1, bias=False)]
    if bn:
        block_list += [BatchNorm(width)]
    block_list += [nn.LeakyReLU(.2, inplace=True)]

    return nn.Sequential(*block_list)


class ResNetBottleneck(nn.Module):
    """
    Args:
        Conv        nn.Conv2d or nn.Conv3d 
        width       width of bottleneck
        stride      convolution stride (int or tuple of ints)
        noise       boolen flag: use Noise layer or do not 
        sigma       standard deviation of the gaussian noise
                    used in Noise layer
        bn          whether to add BatchNorm layer
    """
    def __init__(
            self, Conv, in_channels, out_channels, stride=1, 
            width=None, noise=False, sigma=None, bn=True):
        super().__init__()

        self.proj = None
        if ((torch.tensor(stride) > 1).any() or 
                in_channels != out_channels):
            self.proj = Conv(in_channels, out_channels, 1, stride)
            
        if not width:
            width = (in_channels + out_channels) // 4

        self.main = nn.Sequential (
            block1x1(Conv, in_channels, width, noise, sigma, bn),
            block3x3(Conv, width, stride, noise, sigma, bn),
            block1x1(Conv, width, out_channels, noise, sigma, bn)
        )
        self.leaky = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        y = self.main(x)
        if self.proj is not None:
            x = self.proj(x)
            
        return self.leaky(y + x)


class ImageDiscriminator(nn.Module):
    def __init__(
            self, in_channels=3, cond_size=64,
            base_width=32, noise=False, sigma=None):
        super().__init__()
        
        # ResNetBottlneck 'partial class'
        ResNetBlock = lambda inC, outC, stride: ResNetBottleneck(
            nn.Conv2d, inC, outC, stride, noise=noise, sigma=sigma)

        self.D1 = nn.Sequential (
            nn.Conv2d(in_channels, base_width, 1, 2), 
            ResNetBlock(base_width, base_width*2, 2),
            ResNetBlock(base_width*2, base_width*4, 2),
            ResNetBlock(base_width*4, base_width*8, 2)
        )
        # << output size: (-1, base_width*8, 4, 4)

        cat_dim = base_width*8 + cond_size
        self.D2 = nn.Sequential (
            ResNetBlock(cat_dim, cat_dim, 1),
            nn.Conv2d(cat_dim, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, input, condition):
        out = self.D1(input)
        # >> original shape of condition: (-1, cond_size)
        condition = condition[..., None, None].expand(*condition.shape, 4, 4)
        # << after taking a slice: (-1, cond_size, 1, 1);
        #    after expand: (-1, cond_size, 4, 4)
        out = torch.cat((out, condition), 1)

        return self.D2(out)


class VideoDiscriminator(nn.Module):
    def __init__(
            self, in_channels=3, cond_shape=(16, 64),
            base_width=32, noise=False, sigma=None):
        super().__init__()

        # ResNetBottlneck 'partial class' 
        ResNetBlock = lambda inC, outC, stride: ResNetBottleneck(
            nn.Conv3d, inC, outC, stride, noise=noise, sigma=sigma)#, bn=False)

        self.D1 = nn.Sequential (
            nn.Conv3d(in_channels, base_width, 1), 
            ResNetBlock(base_width, base_width*2, (1,2,2)),
            ResNetBlock(base_width*2, base_width*4, (1,2,2)),
        )
        # << ouput size: (-1, base_width*8, cond_shape[0], 16, 16)

        cat_dim = base_width*4 + cond_shape[1]
        self.D2 = nn.Sequential (
            ResNetBlock(cat_dim, cat_dim, 2),
            ResNetBlock(cat_dim, base_width*8, 2),
            nn.Conv3d(base_width*8, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, input, condition):
        out = self.D1(input)
        condition = condition.expand(16, 16, *condition.shape)
        # << condition.shape: (16, 16, -1, *cond_shape) 
        condition = condition.permute(2, 4, 3, 0, 1)
        # << condition.shape: (-1, cond_shape[::-1], 16, 16)
        out = torch.cat((out, condition), 1)
        
        return self.D2(out)


class VideoGenerator(nn.Module):
    """
    Args:
        dim_Z           noise dimensionality
        cond_shape      2-tuple:  (r, u)
                        r - number of text features 
                        u - their emb. size 
    """
    def __init__(
            self, dim_Z, cond_shape=(16, 64), 
            n_colors=3, base_width=64, video_length=16):
        super().__init__()
        self.dim_Z = dim_Z
        self.n_colors = n_colors
        self.vlen = video_length
        self.code_size = dim_Z + cond_shape[1]

        self.gru = nn.GRU(
                self.code_size, self.code_size, batch_first=True)

        ResNetBlock = lambda inC, outC, stride: ResNetBottleneck(
                nn.Conv2d, inC, outC, stride)

        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ResNetBlock(self.code_size, base_width*8, 1),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            ResNetBlock(base_width*8, base_width*8, 1),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            ResNetBlock(base_width*8, base_width*4, 1),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            ResNetBlock(base_width*4, base_width*2, 1),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            ResNetBlock(base_width*2, base_width, 1),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(base_width, self.n_colors, 3, 1, 1),

            nn.Tanh()
        )

    def forward(self, condition, vlen=None):
        vlen = vlen if vlen else self.vlen

        code = condition.new(
            len(condition), vlen, self.code_size).normal_()
        initial = code.new(
            1, len(condition), self.code_size).normal_()

        code[..., self.dim_Z:] = condition

        H,_ = self.gru(code, initial)
        H = torch.flatten(H, 0, 1)[..., None, None]
        out = self.main(H).view(-1, vlen, 3, 64, 64)

        return out.permute(0, 2, 1, 3, 4)
