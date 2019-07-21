import torch
import torch.nn as nn
import torch.utils.data

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
        emb_weights     matrix with embedding in its rows

        set of the rest ('extra') tuning hyperparameters

        hyppar[0]       hyperparameter for attention part
        hyppar[1]       number of lstm hidden units 
                        (affects the size of output matrix 'M')
        hyppar[2]       hyperparameter for small "cnn" 
                        (affects the size of the ouput matrix 'C')
    """
    def __init__(
            self, n_spots, emb_weights, hyppar=(64,64,64)):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained (
                        emb_weights, 
                        freeze=False,
                        padding_idx=0 
                    )
        emb_size = emb_weights.size(1)

        self.attention = nn.Sequential (
            nn.Linear(hyppar[1]*2, hyppar[0]),
            nn.Tanh(),
            nn.Linear(hyppar[0], n_spots),
            nn.Softmax(1)
        )
        self.lstm = nn.LSTM (
            emb_size, hyppar[1], 
            batch_first=True, bidirectional=True
        )

        self.cnn = nn.Sequential (
            nn.Conv1d(emb_size, hyppar[2]*2, 5, 1, 2),
            nn.LeakyReLU(.2, True),
            nn.MaxPool1d(2),
            nn.Conv1d(hyppar[2]*2, hyppar[2]*4, 3, 1, 1),
            nn.LeakyReLU(.2, True),
            nn.MaxPool1d(2),
            nn.Conv1d(hyppar[2]*4, hyppar[2]*4, 3, 1, 1)
        )

        self.sp = hyppar[2] * 2   # split index

    def forward(self, text_ids):
        E = self.embed(text_ids)
        H = self.lstm(E)[0]

        A = self.attention(H)
        # << batch_size x sen_len x n_spots
        M = torch.einsum('ikp,ikq->ipq', A, H)
        # << batch_size x n_spots x (hyppar[1]*2)

        H = self.cnn(E.permute(0, 2, 1))
        C = H[:, :self.sp] * torch.sigmoid(H[:, self.sp:])
        # << batch_size x ceil(ceil(sen_len / 2) / 2) x (hyppar[2]*2)

        return (M, C), A



class ResNetBottleneck(nn.Module):
    """
    Args:
        type        2d or 3d data
        width       width of bottleneck
        stride      convolution stride (int or tuple of ints)
        noise       boolen flag: use Noise layer or do not 
        sigma       standard deviation of the gaussian noise
                    used in Noise layer
    """
    def __init__(
            self, type, in_channels, out_channels,
            stride=1, width=None, noise=False, sigma=None):
        super().__init__()

        if type == '3d':
            Convolution = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
        elif type == '2d':
            Convolution = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
        else:
            raise TypeError (
                "__init__(): argument 'type' "
                "must be '2d' or '3d'"
            )

        self.proj = None
        if ((torch.tensor(stride) > 1).any() or 
                in_channels != out_channels):
            self.proj = Convolution(
                    in_channels, out_channels, 1, stride)
            
        if not width:
            width = (in_channels + out_channels) // 4

        self.main = nn.Sequential (
            Noise(noise, sigma=sigma),
            Convolution(in_channels, width, 1, bias=False),
            BatchNorm(width),
            nn.LeakyReLU(.2, True),

            Noise(noise, sigma=sigma),
            Convolution(width, width, 3, stride, 1, bias=False),
            BatchNorm(width),
            nn.LeakyReLU(.2, True),

            Noise(noise, sigma=sigma),
            Convolution(width, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.LeakyReLU(.2, True),
        )

    def forward(self, x):
        y = self.main(x)
        if self.proj is not None:
            x = self.proj(x)
            
        return y + x


class ImageDiscriminator(nn.Module):
    def __init__(
            self, in_channels=3, cond_shape=(5, 128),
            base_width=32, noise=False, sigma=None):
        super().__init__()
        
        # form filters' shape
        self.mixer = nn.Parameter(torch.Tensor(2, cond_shape[0]))
        nn.init.kaiming_uniform_(self.mixer, a=(5**.5))
        self.dense_shaper = nn.Linear(cond_shape[1], 128) 

        # 'partial class' of ResNetBottlneck
        ResNetBlock = lambda inC, outC, stride: ResNetBottleneck(
                '2d', inC, outC, stride, noise=noise, sigma=sigma)

        # stacked discriminator components
        self.D1 = nn.Sequential (
            nn.Conv2d(in_channels, base_width, 1), 
            ResNetBlock(base_width, base_width*2, 2)
        )
        self.D2 = nn.Sequential (
            ResNetBlock(base_width*2, base_width*4, 2),
            ResNetBlock(base_width*4, base_width*4, 2)
        )
        self.D3 = nn.Sequential (
            ResNetBlock(base_width*4, base_width*4, 2),
            ResNetBlock(base_width*4, base_width*8, 2)
        )

        # 1x1 convolutions and fc layers to obtain desired shape
        self.conv_shaper1 = nn.Conv2d(base_width*2, 4, 1, 2)
        self.conv_shaper2 = nn.Conv2d(base_width*4, 16, 1)

        self.dense_shaper1 = nn.Linear(4*16*16, 128)
        self.dense_shaper2 = nn.Linear(16*8*8, 128)
        self.dense_shaper3 = nn.Linear(256*4, 128)
        
        self.leaky = nn.LeakyReLU(.2, True)

    def forward(self, input, condition):
        interim = self.mixer @ condition
        # << batch_size x 2 x cond_shape[1]
        filters = self.dense_shaper(self.leaky(interim))
        # << batch_size x 2 x 128
        
        out1 = self.D1(input)
        out2 = self.D2(out1)
        out3 = self.D3(out2)
        
        out1 = self.conv_shaper1(out1)
        out2 = self.conv_shaper2(out2)

        out1 = self.dense_shaper1(out1.view(-1,1024)) 
        out2 = self.dense_shaper2(out2.view(-1,1024)) 
        out3 = self.dense_shaper3(out3.view(-1,1024))
        
        out1 = filters[:, 0] * out1
        out2 = filters[:, 0] * out2

        return torch.cat((out1,out2,out3), 1)



class VideoDiscriminator(nn.Module):
    def __init__(
            self, in_channels=3, cond_shape=(8, 128),
            base_width=32, noise=False, sigma=None):
        super().__init__()

        self.mixer = nn.Parameter(torch.Tensor(3, cond_shape[0]))
        nn.init.kaiming_uniform_(self.mixer, a=(5**.5))

        # fc layers to form desired shapes
        self.dense_shaper11 = nn.Linear(cond_shape[1], 8*8*3*5*5) 
        self.dense_shaper12 = nn.Linear(cond_shape[1], 16*16*3*3) 
        self.dense_shaper13 = nn.Linear(cond_shape[1], 32*32*3) 

        # 'partial class' of ResNetBottlneck
        ResNetBlock = lambda inC, outC, stride: ResNetBottleneck(
                '3d', inC, outC, stride, noise=noise, sigma=sigma)

        # stacked discriminator components
        self.D1 = nn.Sequential (
            nn.Conv3d(in_channels, base_width, 1), 
            ResNetBlock(base_width, base_width*2, 2)
        )
        self.D2 = nn.Sequential (
            ResNetBlock(base_width*2, base_width*4, 2),
            ResNetBlock(base_width*4, base_width*4, (1,2,2))
        )
        self.D3 = nn.Sequential (
            ResNetBlock(base_width*4, base_width*4, 2),
            ResNetBlock(base_width*4, base_width*8, 2)
        )

        # 1x1 convolutions to obtain desired shapes
        self.conv_shaper1 = nn.Conv3d(base_width*2, 8, 1)
        self.conv_shaper2 = nn.Conv3d(base_width*4, 16, 1)
        self.conv_shaper3 = nn.Conv3d(base_width*8, 32, 1) 
        
        # post processing of D-outs convolved with filters 
        self.processor1 = nn.Sequential (
            nn.Conv3d(8, 8, 3, 2, 1),
            nn.LeakyReLU(.2, True),
            nn.Conv3d(8, 8, 3, 2, 1),
            nn.LeakyReLU(.2, True)
        )
        self.processor2 = nn.Sequential (
            nn.Conv3d(16, 16, 3, 2, 1),
            nn.LeakyReLU(.2, True)
        )

        # mix incoming neurons
        self.dense_shaper21 = nn.Linear(8*4*4, 128)
        self.dense_shaper22 = nn.Linear(16*2*2*2, 128)
        self.dense_shaper23 = nn.Linear(32*2*2, 128)
        
        self.leaky = nn.LeakyReLU(.2, True)

    def forward(self, input, condition):
        interim = self.mixer @ condition
        # << batch_size x 3 x cond_shape[1]

        filter1 = self.dense_shaper11(self.leaky(interim[:, 0]))
        filter2 = self.dense_shaper12(self.leaky(interim[:, 1]))
        filter3 = self.dense_shaper13(self.leaky(interim[:, 2]))
        # << filter.shape:  batch_size x dense_shaper.out_features

        out1 = self.D1(input)
        out2 = self.D2(out1)
        out3 = self.D3(out2)

        # >> batch size
        N = input.size(0)

        out1 = torch.conv3d(
                self.conv_shaper1(out1).view(1,-1,8,32,32),
                filter1.view(-1,8,3,5,5), stride=2, 
                padding=(1,2,2), groups=N).view(-1,8,4,16,16)
        out2 = torch.conv3d(
                self.conv_shaper2(out2).view(1,-1,4,8,8), 
                filter2.view(-1,16,1,3,3), stride=(1,2,2), 
                padding=(0,1,1), groups=N).view(-1,16,4,4,4)
        out3 = torch.conv1d(
                self.conv_shaper3(out3).view(1,-1,4), 
                filter3.view(-1,32,3), stride=1,
                padding=1, groups=N).view(-1,32*4)

        out1 = torch.flatten(self.processor1(out1), 1)
        out2 = torch.flatten(self.processor2(out2), 1)

        out1 = self.dense_shaper21(out1)
        out2 = self.dense_shaper22(out2)
        out3 = self.dense_shaper23(out3)

        return torch.cat((out1,out2,out3), 1)



class VideoGenerator(nn.Module):
    def __init__(
            self, dim_Z, cond_shape, 
            n_colors=3, base_width=32, video_length=16):
        super().__init__()
        self.dim_Z = dim_Z
        self.n_colors = n_colors
        self.vlen = video_length
        self.code_size = dim_Z + cond_shape[1]

        self.mixer = nn.Parameter(
                torch.Tensor(self.vlen, cond_shape[0]))
        nn.init.kaiming_uniform_(self.mixer, a=(5**.5))

        self.gru = nn.GRU(
                self.code_size, self.code_size, batch_first=True)

        self.main = nn.Sequential (
            ResNetBottleneck('3d', self.code_size, base_width*8),
            nn.Upsample(scale_factor=2, mode='trilinear'),

            ResNetBottleneck('3d', base_width*8, base_width*4, (2,1,1)),
            nn.Upsample(scale_factor=2, mode='trilinear'),

            ResNetBottleneck('3d', base_width*4, base_width*4, (2,1,1)),
            nn.Upsample(scale_factor=2, mode='trilinear'),

            ResNetBottleneck('3d', base_width*4, base_width*2, (2,1,1)),
            nn.Upsample(scale_factor=2, mode='trilinear'),

            ResNetBottleneck('3d', base_width*2, base_width, (2,1,1)),
            nn.Upsample(scale_factor=2, mode='trilinear'),

            ResNetBottleneck('3d', base_width, base_width, (2,1,1)),
            nn.Upsample(scale_factor=2, mode='trilinear'),

            nn.Conv3d(base_width, self.n_colors, 3, (2,1,1), 1),
            nn.Tanh()
        )

    def forward(self, text_features, vlen=None):
        vlen = vlen if vlen else self.vlen

        # >> mixer matrix is used here just to create 'code' 
        # >> tensor on the right gpu node with .new() method
        code = self.mixer.new(
            len(text_features), vlen+1, self.code_size).normal_()

        condition = torch.einsum(
                'ink,mn->imk', text_features, self.mixer)

        code[:, 1:, self.dim_Z:] = condition

        H = self.gru(code[:, 1:], code[None, :, 0])[0] \
                .permute(0, 2, 1)[..., None, None]

        return self.main(H)
