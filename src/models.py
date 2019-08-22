import torch
import torch.nn as nn
import torch.utils.data

from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from resnet import BasicBlock


class SimpleTextEncoder:
    """
    Embedding of a sentence is obtained as the average
    of pretrained GloVe embeddings of the words
    that make up the sentence
    """
    def __init__(self, emb_weights):
        self.embed = nn.Embedding.from_pretrained(
                emb_weights, padding_idx=0)

    def __call__(self, text_ids, lengths):
        return self.embed(text_ids).sum(1) / lengths[:, None].float()

    def to(self, device):
        self.embed = self.embed.to(device)


class TextEncoder(nn.Module):
    """
    Args:
        emb_weights     matrix of size (n_tokens, emb_dim) 
        proj            project to lower dimension space
        train_embs      whether to train embeddings or not
                        (by default, the value is False,
                         i.e. emb_weights are frozen)

        hyppar[0]       number of gru hidden units
        hyppar[1]       projection dimensionality
    """
    def __init__(
            self, emb_weights, proj=False,
            train_embs=False, hyppar=(64,64)):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained (
                        emb_weights,
                        freeze=(not train_embs),
                        padding_idx=0 
                    )
        self.gru = nn.GRU (
            emb_weights.size(1), hyppar[0],
            batch_first=True, bidirectional=True
        )
        if proj:
            self.proj = nn.Sequential (
                nn.Linear(hyppar[0]*2, hyppar[1]),
                nn.LeakyReLU(.2, inplace=True)
            )
        else: 
            self.proj = lambda x: x

    def forward(self, text_ids, lengths):
        lengths, sortbylen = lengths.sort(0, descending=True)
        H = self.embed(text_ids[sortbylen])
        H = pack_padded_sequence(H, lengths, batch_first=True)

        _, last = self.gru(H)
        out = self.proj(torch.cat(tuple(last), 1))
        _, unsort = sortbylen.sort(0)

        return out[unsort]


class ImageDiscriminator(nn.Module):
    """
    Args:
        k   number of selected frames
    """
    def __init__(
            self, in_channels=3, cond_size=64, k=8,
            base_width=32, noise=False, sigma=.2):
        super().__init__()
        self.k = k
        
        # Bottleneck 'partial class'
        ResNetBlock = lambda inC, outC, stride: BasicBlock(
            '2d', inC, outC, stride, bn=False, noise=noise, sigma=sigma)

        self.D1 = nn.Sequential (
            ResNetBlock(in_channels, base_width, 2), 
            ResNetBlock(base_width, base_width*2, 2),
            ResNetBlock(base_width*2, base_width*4, 2),
        )
        # << output size: (-1, base_width*8, 4, 4)

        cat_dim = base_width*4 + cond_size
        self.D2 = nn.Sequential (
            ResNetBlock(cat_dim, cat_dim*2, 1),
            nn.Conv2d(cat_dim*2, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        """
        x: video
        c: condition (do not confuse
           with C - number of filters)
        """
        selected_frames = torch.multinomial(
                torch.ones(x.size(2)), self.k)
        # >> x.shape: (-1, C, D, H, W)
        x = x[:, :, selected_frames, ...]
        # << x.shape: (-1, C, k, H, W)
        x = torch.flatten(x.permute(0, 2, 1, 3, 4), 0, 1)
        # << x.shape: (-1, C, H, W)

        x = self.D1(x)
        c = c[None, ..., None, None].expand(self.k, *c.shape, 4, 4)
        x = torch.cat((x, torch.flatten(c,0,1)), 1)

        return self.D2(x)


class VideoDiscriminator(nn.Module):
    def __init__(
            self, in_channels=3, cond_size=64,
            base_width=32, noise=False, sigma=.2):
        super().__init__()

        # BasicBlock 'partial class'
        ResNetBlock = lambda inC, outC, stride: BasicBlock(
            '3d', inC, outC, stride, bn=False, noise=noise, sigma=sigma)

        self.D1 = nn.Sequential (
            ResNetBlock(in_channels, base_width, 2), 
            ResNetBlock(base_width, base_width*2, 2),
            ResNetBlock(base_width*2, base_width*4, 2),
            ResNetBlock(base_width*4, base_width*8, (2,1,1)),
        )
        # << ouput size: (-1, base_width*8, 1, 4, 4)

        cat_dim = base_width*8 + cond_size
        self.D2 = nn.Sequential (
            ResNetBlock(cat_dim, cat_dim*2, 1),
            nn.Conv3d(cat_dim*2, 1, (1, 4, 4)),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        """
        x: video
        c: condition
        """
        x = self.D1(x)
        c = c.view(*c.shape,1,1,1).expand(*c.shape,1,4,4)
        x = torch.cat((x, c), 1)
        
        return self.D2(x)


class VideoGenerator(nn.Module):
    """
    Args:
        dim_Z           noise dimensionality
        cond_size       condition size
    """
    def __init__(
            self, dim_Z, cond_size=64, 
            n_colors=3, base_width=128, video_length=16):
        super().__init__()
        self.dim_Z = dim_Z
        self.n_colors = n_colors
        self.vlen = video_length
        self.code_size = dim_Z + cond_size

        self.gru = nn.GRU(
                self.code_size, self.code_size, batch_first=True)

        ResNetBlock = lambda inC, outC, stride: BasicBlock(
                '3d', inC, outC, stride)

        self.main = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2)),
            ResNetBlock(self.code_size, base_width*8, 1),

            nn.Upsample(scale_factor=(1,2,2)),
            ResNetBlock(base_width*8, base_width*4, 1),

            nn.Upsample(scale_factor=(1,2,2)),
            ResNetBlock(base_width*4, base_width*2, 1),

            nn.Upsample(scale_factor=(1,2,2)),
            ResNetBlock(base_width*2, base_width, 1),

            nn.Upsample(scale_factor=(1,2,2)),
            nn.Conv3d(base_width, self.n_colors, 3, 1, 1),

            nn.Tanh()
        )

    def forward(self, c, vlen=None):
        """
        c: condition (batch_size, cond_size)
        vlen: video length
        """
        vlen = vlen if vlen else self.vlen

        code = c.new(len(c), vlen, self.code_size).normal_()
        code[..., self.dim_Z:] = c[:, None, :]

        H,_ = self.gru(code)
        H = H.permute(0, 2, 1)[..., None, None]

        return self.main(H)
