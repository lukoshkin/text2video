import torch
import torch.nn as nn
import torch.utils.data

from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from torch.nn.utils import spectral_norm as SN
from functools import partial
from blocks import DBlock, GBlock, CGBlock
from convgru import ConvGRU


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


class StackImageDiscriminator(nn.Module):
    """
    Args:
        k   number of selected frames
    """
    def __init__(
            self, in_channels=3, cond_size=64, k=8,
            base_width=32, noise=False, sigma=.2):
        super().__init__()
        self.k = k
        self.D1 = nn.Sequential (
            SN(nn.Conv2d(in_channels, base_width, 1)),
            DBlock('2d', base_width, base_width, 2),
            DBlock('2d', base_width, base_width*2, 2),
            DBlock('2d', base_width*2, base_width*4, 2),
        )
        # << output size: (-1, base_width*4, 4, 4)

        cat_dim = base_width*4 + cond_size
        self.D2 = nn.Sequential (
            DBlock('2d', cat_dim, cat_dim*2, 1),
            SN(nn.Conv2d(cat_dim*2, 1, 4)),
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


class StackVideoDiscriminator(nn.Module):
    def __init__(
            self, in_channels=3, cond_size=64,
            base_width=32, noise=False, sigma=.2):
        super().__init__()
        self.D1 = nn.Sequential (
            SN(nn.Conv3d(in_channels, base_width, 1)),
            DBlock('3d', base_width, base_width, 2),
            DBlock('3d', base_width, base_width*2, 2),
            DBlock('3d', base_width*2, base_width*4, 2),
            DBlock('3d', base_width*4, base_width*8, (2,1,1)),
        )
        # << ouput size: (-1, base_width*4, 1, 4, 4)

        cat_dim = base_width*8 + cond_size
        self.D2 = nn.Sequential (
            DBlock('3d', cat_dim, cat_dim*2, 1),
            SN(nn.Conv3d(cat_dim*2, 1, (1, 4, 4))),
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


class SimpleVideoGenerator(nn.Module):
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

        GB = partial(GBlock, '3d', stride=(1,2,2))

        self.main = nn.Sequential(
            GB(self.code_size, base_width*8),
            GB(base_width*8, base_width*4),
            GB(base_width*4, base_width*2),
            GB(base_width*2, base_width),
            GB(base_width, self.n_colors),
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


class TestVideoGenerator(nn.Module):
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
        
        CGB = partial(CGBlock, '3d', self.code_size, stride=(1,2,2))

        self.gblock1 = CGB(self.code_size, base_width*8)
        self.gblock2 = CGB(base_width*8, base_width*4)
        self.gblock3 = CGB(base_width*4, base_width*2)
        self.cgru = ConvGRU(
                base_width*2, base_width*2, 3, spectral_norm=True)
        self.gblock4 = CGB(base_width*2, base_width)
        self.gblock5 = CGB(base_width, self.n_colors)

    def forward(self, c, vlen=None):
        """
        c: condition (batch_size, cond_size)
        vlen: video length
        """
        vlen = vlen if vlen else self.vlen

        code = c.new(len(c), vlen, self.code_size).normal_()
        code[..., self.dim_Z:] = c[:, None, :]

        vcon = c.new(len(c), self.code_size).normal_()
        vcon[:, self.dim_Z:] = c

        H,_ = self.gru(code)
        H = H.permute(0, 2, 1)[..., None, None]

        H = self.gblock1(H, vcon)
        H = self.gblock2(H, vcon)
        H = self.gblock3(H, vcon)
        H,_ = self.cgru(H)
        H = self.gblock4(H, vcon)
        H = self.gblock5(H, vcon)

        return torch.tanh(H)
