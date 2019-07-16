import torch
import torch.nn as nn
import torch.utils.data

device = torch.device (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super().__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * torch.randn_like(x)
        return x



class TextEncoder(nn.Module):
    def __init__(self, n_tokens, n_spots, 
                 emb_weights, hyppar=64):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained (
                        emb_weights, 
                        freeze=False,
                        padding_idx=0 
                    )
        lstm_units = emb_weights.size(1)
        self.attention = nn.Sequential (
                nn.Linear(2 * lstm_units, hyppar),
                nn.Tanh(),
                nn.Linear(hyppar, n_spots),
                nn.Softmax(1)
            )
        self.lstm = nn.LSTM (
            n_tokens, lstm_units, 
            batch_first=True, bidirectional=True
        )

    def forward(self, text_ids):
        H = self.embed(text_ids)
        H = self.lstm(H)[0]

        A = self.attention(H)
        M = torch.einsum('ikp,ikq->ipq', A, H)

        S = torch.softmax(H, 1)
        c = torch.einsum('ikp,ikq->iq', S, H) / n_spots

        return (M, c), A



class ImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, 
                 use_noise=False, noise_sigma=None):
        super().__init__()

        self.main = nn.Sequential (
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input, condition):
        return self.main(input).squeeze()



class VideoDiscriminator(nn.Module):
    def __init__ (
            self, n_channels, n_output_neurons=1, ndf=64,
            use_noise=False, noise_sigma=None):

        super().__init__()
        self.n_output_neurons = n_output_neurons

        self.main = nn.Sequential (
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(
                n_channels, ndf, 4, (1, 2, 2), (0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(
                ndf, ndf * 2, 4, (1, 2, 2), (0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(
                ndf * 2, ndf * 4, 4, (1, 2, 2), (0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(
                ndf * 4, ndf * 8, 4, (1, 2, 2), (0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 8, n_output_neurons, 4, 1, 0, bias=False),
        )

    def forward(self, input, condition):
        return self.main(input).squeeze()



class VideoGenerator(nn.Module):
    def __init__(self, n_channels, ngf=64, 
                 dim_zC, dim_zM, dim_Cond=0, video_length):
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
                   .view(n_samples, vlen, self.inc, *h.size()[3:]) # h is not defined
                   .permute(0, 2, 1, 3, 4)
