import torch
import torch.nn as nn
import torch.utils.data

device = torch.device (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

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
    def __init__(
            self, n_spots, emb_weights, hyppar=64):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained (
                        emb_weights, 
                        freeze=False,
                        padding_idx=0 
                    )
        emb_size = emb_weights.size(1)
        self.sp = emb_size

        self.attention = nn.Sequential (
                nn.Linear(emb_size * 2, hyppar),
                nn.Tanh(),
                nn.Linear(hyppar, n_spots),
                nn.Softmax(1)
            )
        self.lstm = nn.LSTM (
            emb_size, emb_size, 
            batch_first=True, bidirectional=True
        )

        self.cnn = nn.Sequential (
            nn.Conv1d(emb_size, emb_size * 2, 5, 1, 2)
            nn.LeakyReLU(0.2, True)
            nn.MaxPool1d(2)
            nn.Conv1d(emb_size * 2, emb_size * 4, 3, 1, 1)
            nn.LeakyReLU(0.2, True)
            nn.MaxPool1d(2)
            nn.Conv1d(emb_size * 4, emb_size * 4, 3, 1, 1)
        )

    def forward(self, text_ids):
        E = self.embed(text_ids)
        H = self.lstm(E)[0]

        A = self.attention(H)
        # batch_size x sen_len x n_spots
        M = torch.einsum('ikp,ikq->ipq', A, H)
        # batch_size x n_spots x (emb_size * 2)

        H = self.cnn(E)
        C = H[:, :self.sp] * torch.sigmoid(H[:, self.sp:]
        # batch_size x ceil(ceil(sen_len / 2) / 2) x (emb_size * 2)

        return (M, C), A



class ResNetBlockLike(nn.Module):
    def __init__(
            self, type, in_channels, out_channels, 
            kernel_size=4, stride=None, padding=None, 
            noise=False, sigma=None):
        super().__init__()

        # default arguments
        if stride is None:
            stride = (1, 2, 2) if type == 'video' else 2
        if padding is None:
            padding = (0, 1, 1) if type == 'video' else 1

        if type == 'video':
            Convolution = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
        elif type == 'image':
            Convolution = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
        else:
            raise TypeError (
                "__init__(): argument 'type' " 
                "must be 'video' or 'image'"
            )

        self.proj = None
        if in_channels != out_channels:
            self.proj = Convolution(
                in_channels, out_channels, 1, stride)

        self.main = nn.Sequential (
            Noise(noise, sigma=sigma),
            Convolution(
                in_channels, out_channels, 
                kernel_size, stride, 
                padding, bias=False),
            BatchNorm(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        y = self.main(x)
        if self.proj is not None:
            x = self.proj(x)

        return y + x



class TypeDiscriminator(nn.Module):
    def __init__(
            self, type, in_channels, cond_size, emb_size,
            base_width=32, noise=False, sigma=None):
        super().__init__()

        self.mixer = nn.Parameter(torch.Tensor(3, cond_size))
        #nn.init.xavier_normal_(self.mixer)
        self.dense_shaper1 = nn.Linear(emb_size, 8 * 8 * 5 * 5)

        self.Conv = nn.Conv3d if type == 'video' else nn.Conv2d

        self.D1 = nn.Sequential (
            self.Conv(in_channels, base_width, 1), 
            ResNetBlockLike(type, base_width, base_width * 2),
            ResNetBlockLike(type, base_width * 2, base_width * 4)
        )
        self.D2 = nn.Sequential (
            ResNetBlockLike(type, base_width * 4, base_width * 4),
            ResNetBlockLike(type, base_width * 4, base_width * 4),
        )
        self.D3 = nn.Sequential (
            ResNetBlockLike(type, base_width * 4, base_width * 8),
            ResNetBlockLike(type, base_width * 8, base_width * 8),
        )

        self.conv_shaper1 = nn.Sequential (
            self.Conv(base_width * 4, 8),
            # not implemented
            nn.AvgPool2d(2)
        )
        self.conv_shaper2 = nn.Sequential (
            self.Conv(base_width * 4, 8),
            # not implemented
        )
        self.dense_shaper2 = nn.Linear(calc_in_feats, 128)

    def forward(self, input, condition):
        interim = torch.einsum('ink,mn->imk', condition, self.mixer)
        filters = torch.relu(self.dense_shaper1(interim))

        temp1 = self.D1(input)
        temp2 = self.D2(temp1)
        out1 = self.conv_shaper1(temp1).view(-1, 128)
        out2 = self.conv_shaper2(temp2).view(-1, 128)
        out3 = self.dense_shaper2(self.D3(temp2))

        # not implemented 

        return 



class VideoGenerator(nn.Module):
    def __init__(
            self, n_channels, dim_zC, dim_zM, 
            dim_Cond=0, ngf=64, video_length=16):
        super().__init__()

        self.inc = n_channels # in-colors
        self.dim_zC = dim_zC
        self.dim_zM = dim_zM
        self.vlen = video_length
        self.code_dims = {'image' : dim_zC + dim_Cond,
                          'video' : dim_zM + dim_Cond}

        if dim_Cond:
            self.fc = nn.Linear(dim_Cond, dim_Cond)

        self.RNN = nn.GRUCell(dim_zM, dim_zM)

        dim_Z = dim_zM + dim_zC + dim_Cond
        self.main = nn.Sequential (
            nn.ConvTranspose2d(dim_Z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, self.inc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def sample_zM(self, n_samples, condition, vlen=None):
        vlen = vlen if vlen else self.vlen

        emb_size = self.code_dims['video']
        code = torch.randn (
                    vlen + 1, n_samples, emb_size, 
                    device=device
                )
        if condition is not None:
            code[:, self.dim_zM:] = condition

        h = [code[-1]]
        for i in range(vlen):
            h.append(self.RNN(code[i], h[-1]))

        return torch.stack(h[1:], dim=1) \
                    .view(-1, emb_size)

    def sample_zC(self, n_samples, condition, vlen=None):
        vlen = vlen if vlen else self.vlen

        emb_size = self.code_dims['image']
        code = torch.randn (
                    n_samples, emb_size, 
                    device=device
                )
        if condition is not None:
            code[:, self.dim_zC:] = condition

        return code.repeat(1, vlen) \
                   .view(-1, emb_size)

    def sample_Z(self, n_samples, conditions, vlen=None):
        at_video, at_image= None, None

        if conditions is not None:
            at_video, at_image = conditions
            at_video = self.mixer(at_video)

        zC = self.sample_zC(n_samples, at_image, vlen)
        zM = self.sample_zM(n_samples, at_video, vlen)

        return torch.cat([zC, zM], dim=1)

    def sample_videos(self, n_samples, conditions, vlen=None):
        vlen = vlen if vlen else self.vlen
        z = self.sample_Z(n_samples, conditions, vlen)

        return self.main(z.view(*z.size(), 1, 1)) \
                   .view(n_samples, vlen, self.inc, *h.size()[3:]) \
                   .permute(0, 2, 1, 3, 4)
